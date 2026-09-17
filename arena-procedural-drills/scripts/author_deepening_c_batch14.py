#!/usr/bin/env python3
"""Author 8 ex3 deepening drills (batch 14, group C — prereqs_autograd_internals).

Each ex3 hits a DISTINCT third facet vs ex1/ex2. ONE LO + ONE Bloom + <=2 KCs.

Atoms (all under prereqs_autograd_internals):
    - arg-position-back-functions   (ex3: BACK_FUNCS registry/dispatch by (fn, argnum))
    - chain-rule-elementwise        (ex3: parameterized leaky_relu_back — slope kwarg)
    - grad-tracking-global-toggle   (ex3: set_grad_enabled(bool) explicit-value setter, nesting)
    - kwargs-pass-through-recipe    (ex3: replay back-fn via **recipe.kwargs — recover dim from Recipe)
    - parents-dict-by-argidx        (ex3: dispatch back fns over parents.items() — int-keyed positional)
    - recipe-dataclass              (ex3: ternary unary-arg + 2-kwarg case — clamp_forward)
    - requires-grad-propagation     (ex3: chained ops — second wrap inherits requires_grad from first output)
    - unbroadcast-pattern           (ex3: full add_back0 binary-op that uses unbroadcast to restore shape)
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_autograd_internals"


# ---------------------------------------------------------------------------
# Recap blocks
# ---------------------------------------------------------------------------

RECAP_BACK_FUNCS_REGISTRY = (
    "## `BACK_FUNCS` — dispatch back fns by (forward_fn, argnum)\n"
    "\n"
    "Ex1 wrote `div_back0`/`div_back1`; ex2 wrote `pow_back0`/`pow_back1`. The "
    "deepening move is the GLUE: a registry `BACK_FUNCS` keyed by `(fwd_fn, "
    "argnum)` so the reverse pass can ask 'which back fn for the 1st arg of "
    "`t.divide`?' without if/elif chains.\n"
    "\n"
    "```python\n"
    "class BackFuncs:\n"
    "    def __init__(self):\n"
    "        self._registry = {}\n"
    "    def add_back_func(self, fwd_fn, argnum, back_fn):\n"
    "        self._registry[(fwd_fn, argnum)] = back_fn\n"
    "    def get_back_func(self, fwd_fn, argnum):\n"
    "        return self._registry[(fwd_fn, argnum)]\n"
    "```\n"
    "\n"
    "**Why `(fwd_fn, argnum)` as the key.** Each forward fn has ONE back fn "
    "per Tensor input position. `div` registers 2 entries (argnum 0 and 1); "
    "`log` registers 1 (argnum 0 only). The reverse pass iterates "
    "`recipe.parents.items()` (which is `{argnum: parent_tensor}`) and "
    "looks up `BACK_FUNCS.get_back_func(recipe.func, argnum)` for each.\n"
    "\n"
    "**Missing-key error.** If a back fn was never registered, `KeyError` "
    "fires at reverse-pass time. ARENA's MiniTensor uses a custom message "
    "'no back fn registered for `<fn>` at argnum `<i>`' so the failure points "
    "at the missing registration, not the dict internals."
)

RECAP_LEAKY_RELU = (
    "## Parameterized elementwise: `leaky_relu_back` carries the slope through\n"
    "\n"
    "Ex1 (sigmoid/relu) and ex2 (tanh/softplus) used `(grad_out, out, x)` — "
    "no extra parameters. The deepening move is the FIRST parameterized "
    "elementwise op: `leaky_relu(x, negative_slope=0.01)` has TWO regimes.\n"
    "\n"
    "```python\n"
    "# Forward: out = x if x > 0 else negative_slope * x\n"
    "# Derivative: 1 where x > 0, negative_slope where x <= 0.\n"
    "# Backward signature must accept the slope so the reverse pass can\n"
    "# read it back from recipe.kwargs and pass it through.\n"
    "def leaky_relu_back(grad_out, out, x, negative_slope=0.01):\n"
    "    local = t.where(x > 0, t.ones_like(x), t.full_like(x, negative_slope))\n"
    "    return grad_out * local\n"
    "```\n"
    "\n"
    "**Why the kwarg matters at backward time.** Inside the reverse pass, "
    "`leaky_relu_back(grad_out, out, x, **recipe.kwargs)` re-uses the same "
    "slope the forward used. Hard-coding `0.01` would silently produce the "
    "wrong gradient whenever the user changed it. This is the same "
    "kwargs-pass-through pattern from the recipe atoms, applied to a back fn.\n"
    "\n"
    "**Slope = 0 reduces to ReLU.** When `negative_slope == 0`, `local` "
    "is `1` for positive x and `0` for non-positive x — exactly relu's "
    "gradient. This is a useful invariant to test: the deepening drill "
    "checks that the parameterized form REDUCES to the ex1 baseline."
)

RECAP_SET_GRAD_ENABLED = (
    "## `set_grad_enabled(mode)` — explicit-value setter, not just toggle off\n"
    "\n"
    "Ex1 had `NoGrad()` (forces off, restores). Ex2 had `no_grad` decorator "
    "(forces off, restores even on exception). PyTorch ALSO ships "
    "`torch.set_grad_enabled(mode: bool)` — a context manager that sets the "
    "toggle to an EXPLICIT value (`True` OR `False`), and restores the "
    "previous value on exit. This is what you reach for inside a `no_grad` "
    "block when you want to TEMPORARILY re-enable grad for one inner step.\n"
    "\n"
    "```python\n"
    "with set_grad_enabled(False):     # outer: off\n"
    "    assert grad_tracking_enabled is False\n"
    "    with set_grad_enabled(True):  # inner: explicitly on again\n"
    "        assert grad_tracking_enabled is True\n"
    "    assert grad_tracking_enabled is False  # back to outer\n"
    "assert grad_tracking_enabled is True       # back to global default\n"
    "```\n"
    "\n"
    "**Why a value-taking ctx manager is necessary.** `NoGrad` can only force "
    "off. If you've already disabled grad via `no_grad` and want one inner "
    "step to re-enable it (e.g. compute a regularizer gradient during a "
    "no-grad eval loop), you need a setter that takes `True`. Same restore "
    "semantics as `NoGrad`, but the entry value is a parameter.\n"
    "\n"
    "**Stack discipline.** Each `__enter__` snapshots the CURRENT value, "
    "sets the new one. Each `__exit__` restores the snapshot. Arbitrary "
    "nesting works because every frame's snapshot is independent."
)

RECAP_RECIPE_KWARG_REPLAY = (
    "## `**recipe.kwargs` — the back fn replays the forward's keyword args\n"
    "\n"
    "Ex1 threaded kwargs INTO the Recipe; ex2 verified the empty-kwargs "
    "edge case. The deepening move is what the Recipe is FOR: at reverse "
    "time, you call the back fn with `**recipe.kwargs` so the gradient is "
    "computed with the SAME `dim`, `keepdim`, `negative_slope`, etc. the "
    "forward used.\n"
    "\n"
    "```python\n"
    "# Forward: y = sum(x, dim=1, keepdim=True)\n"
    "# Stored: recipe.kwargs == {'dim': 1, 'keepdim': True}\n"
    "# Reverse: grad_in = sum_back(grad_out, y_arr, x_arr, **recipe.kwargs)\n"
    "#         → sum_back receives dim=1, keepdim=True automatically.\n"
    "```\n"
    "\n"
    "**Why this seals the kwargs contract.** Without `**recipe.kwargs` "
    "splatting, the back fn would need to inspect `recipe.kwargs['dim']` "
    "by hand — error-prone and op-specific. Splatting makes the back fn "
    "signature exactly mirror the forward fn signature (minus `grad_out`, "
    "`out`, `x`).\n"
    "\n"
    "**`sum_back` is the canonical example.** `sum` reduces along `dim`. "
    "Its gradient must `unsqueeze` along the same `dim` and `expand` back "
    "to the input shape. Without the dim from kwargs, `sum_back` would "
    "guess — and guess wrong on multi-axis tensors."
)

RECAP_PARENTS_DISPATCH = (
    "## Iterating `parents.items()` to dispatch back fns by argnum\n"
    "\n"
    "Ex1 built `{argnum: tensor}` for positional Tensors; ex2 extended "
    "with kwarg-name keys. The deepening move USES the dict at reverse "
    "time: iterate `parents.items()`, look up `BACK_FUNCS[(fn, argnum)]`, "
    "compute that parent's grad contribution.\n"
    "\n"
    "```python\n"
    "for argnum, parent in recipe.parents.items():\n"
    "    if not isinstance(argnum, int):\n"
    "        continue  # kwarg parents handled separately\n"
    "    back_fn = BACK_FUNCS[(recipe.func, argnum)]\n"
    "    grad_for_parent = back_fn(grad_out, out_arr, *recipe.args)\n"
    "    accumulate(parent, grad_for_parent)\n"
    "```\n"
    "\n"
    "**Why int-only filtering.** ex2 mixed int keys (positional) and str "
    "keys (kwarg names). The dispatcher for POSITIONAL back fns takes the "
    "int keys. Kwarg-positioned Tensors get a SEPARATE back-fn registry "
    "keyed by `(fn, kwarg_name: str)` — different table, same call shape.\n"
    "\n"
    "**`*recipe.args` un-splats stored positional args.** The back fn "
    "signature is `(grad_out, out, *forward_args)`. Storing args as a "
    "tuple and splatting at call time is the cleanest way to feed the "
    "back fn whatever the forward had — `add` has 2 args, `clamp` has 1 "
    "(plus min/max kwargs), `sum` has 1 (plus dim kwarg). Splat handles "
    "all of them uniformly."
)

RECAP_CLAMP_FORWARD = (
    "## Recipe for `clamp(x, min=, max=)` — 1 parent, 2 kwargs\n"
    "\n"
    "Ex1 (log: 1-arg) and ex2 (add: 2-arg) had `kwargs == {}`. The "
    "deepening move populates `recipe.kwargs` for the first time: "
    "`clamp_forward(x, min=lo, max=hi)` has ONE Tensor parent but two "
    "scalar kwargs that the back fn needs to clip at reverse time.\n"
    "\n"
    "```python\n"
    "@dataclass\n"
    "class Recipe:\n"
    "    func: Callable\n"
    "    args: tuple    # → (x.array,)   one positional Tensor\n"
    "    kwargs: dict   # → {'min': -1.0, 'max': 1.0}   non-empty!\n"
    "    parents: dict  # → {0: x}        only positional parent\n"
    "```\n"
    "\n"
    "**Why the kwargs DON'T appear in parents.** `min` and `max` are "
    "scalar floats — non-Tensor inputs are filtered out of `parents` "
    "(same rule as ex1's parents-dict drill). They live ONLY in "
    "`recipe.kwargs` because the back fn needs them, but they have no "
    "gradient flow (you can't take d/dmin of a clamp output).\n"
    "\n"
    "**The reverse pass replay.** `clamp_back(grad_out, out, x, min=..., "
    "max=...)` masks the gradient where `x` was clipped: `grad_in = "
    "grad_out * ((x >= min) & (x <= max)).float()`. The mask is "
    "computed from `recipe.kwargs` — without them stored, the back fn "
    "would have no way to know where the clipping happened."
)

RECAP_REQGRAD_CHAIN = (
    "## requires_grad propagates THROUGH the graph — composition works\n"
    "\n"
    "Ex1 gated `requires_grad` on a single op; ex2 extended the scan to "
    "kwargs. The deepening move tests COMPOSITION: if op A produces a "
    "tensor with `requires_grad=True`, and op B takes that tensor as "
    "input, then B's output also has `requires_grad=True` — even if B's "
    "OTHER inputs are all `requires_grad=False`.\n"
    "\n"
    "```python\n"
    "x = MiniTensor(t.tensor([1.0]), requires_grad=True)\n"
    "y = MiniTensor(t.tensor([2.0]), requires_grad=False)\n"
    "z = MiniTensor(t.tensor([3.0]), requires_grad=False)\n"
    "\n"
    "ab = add(x, y)      # ab.requires_grad == True  (x carries it in)\n"
    "abc = add(ab, z)    # abc.requires_grad == True (ab carries it forward)\n"
    "# The 'rg=True' bit FLOWS down the graph through each op.\n"
    "```\n"
    "\n"
    "**Why this is the propagation invariant.** The three-gate AND in ex1 "
    "decides ONE op's output. Apply that AND at every op in a chain and "
    "you get: as long as `grad_tracking_enabled` and `is_differentiable` "
    "stay True, the `requires_grad` bit propagates from the FIRST "
    "grad-tracked input all the way to the final loss tensor.\n"
    "\n"
    "**Where the propagation STOPS.** If any op in the chain has "
    "`is_differentiable=False` (e.g. `t.equal`, `t.argmax`), the chain "
    "snaps — that op's output has `requires_grad=False` regardless of "
    "input. Downstream ops see only that detached output."
)

RECAP_UNBROADCAST_IN_BACK = (
    "## `add_back0` uses unbroadcast — bridge from unbroadcast to a real back fn\n"
    "\n"
    "Ex1 wrote `unbroadcast` for leading + size-1 axes; ex2 verified the "
    "combined case and idempotence. The deepening move PLUGS unbroadcast "
    "into the smallest real binary back fn: addition.\n"
    "\n"
    "```python\n"
    "# Forward (with broadcasting): out = x + y  where x.shape != y.shape\n"
    "# Mathematical gradient:        dL/dx = grad_out, dL/dy = grad_out\n"
    "# BUT grad_out.shape == out.shape (after broadcast) — it doesn't fit\n"
    "# back into x or y. So each back fn must UNBROADCAST.\n"
    "\n"
    "def add_back0(grad_out, out, x, y):\n"
    "    return unbroadcast(grad_out, x)\n"
    "\n"
    "def add_back1(grad_out, out, x, y):\n"
    "    return unbroadcast(grad_out, y)\n"
    "```\n"
    "\n"
    "**Why every broadcasting binary op needs this.** `add`, `sub`, "
    "`mul`, `div` all broadcast. Their math gradients are simple "
    "(`grad_out` for add, `grad_out * y` for mul, etc.) — but those "
    "shapes match `out`, not the original inputs. Skipping the "
    "unbroadcast step gives gradient tensors of the wrong shape and the "
    "next op in the chain crashes.\n"
    "\n"
    "**Equivalence to torch.autograd.** PyTorch's own AddBackward kernel "
    "does exactly this — sums the incoming grad across axes the input "
    "was broadcast over. We're rebuilding that pattern from scratch.\n"
    "\n"
    "**The exemplar isn't just `add`.** Once `add_back0/1` works, the "
    "same template applies to every elementwise binary op. The ONLY thing "
    "that changes is the math factor (`grad_out`, `grad_out * y`, "
    "`grad_out / y`, ...) BEFORE the unbroadcast call."
)


# ---------------------------------------------------------------------------
# SPEC 1 — arg-position-back-functions ex3 (BACK_FUNCS registry/dispatch)
# ---------------------------------------------------------------------------

SPEC_ARGPOS = {
    "atom_id": "arg-position-back-functions",
    "subtopic": "Backprop: Arg-position back funcs",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_BACK_FUNCS_REGISTRY,
    "exercise_index": 3,
    "exercise_title": "BACK_FUNCS registry and dispatch by (fn, argnum)",
    "slug": "back-funcs-registry-and-dispatch-by-fn-argnum",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["registry", "dispatch", "back-funcs", "argnum"],
    "kcs": [
        "back-funcs-registry-key-is-fn-argnum",
        "missing-back-fn-raises-keyerror",
    ],
    "lo": (
        "Apply the registry-and-dispatch pattern to wire per-(fwd_fn, argnum) "
        "back fns into a `BackFuncs` table, then look them up by the same "
        "key the reverse pass uses."
    ),
    "prompt_body": (
        "Implement `ex3_back_funcs()` returning a class `BackFuncs` and two "
        "back fns wired into an instance of it.\n\n"
        "1. Define class `BackFuncs` with:\n"
        "   - `__init__(self)` initializing `self._registry = {}` (a "
        "dict keyed by `(fwd_fn, argnum)` tuples).\n"
        "   - `add_back_func(self, fwd_fn, argnum, back_fn)` storing "
        "`back_fn` at key `(fwd_fn, argnum)`.\n"
        "   - `get_back_func(self, fwd_fn, argnum)` returning the stored "
        "back fn — raise the natural `KeyError` if missing (don't catch).\n\n"
        "2. Define `div_back0(grad_out, out, x, y)` returning "
        "`grad_out / y` and `div_back1(grad_out, out, x, y)` returning "
        "`-grad_out * out / y`.\n\n"
        "3. Instantiate `bf = BackFuncs()`, register both back fns at the "
        "correct (`t.divide`, `0` | `1`) keys, and return a dict "
        "`{'BackFuncs': BackFuncs, 'bf': bf, 'div_back0': div_back0, "
        "'div_back1': div_back1}`.\n\n"
        "The dispatcher uses `bf.get_back_func(t.divide, 0)` and "
        "`bf.get_back_func(t.divide, 1)` — both must succeed; lookups for "
        "unregistered keys (`t.divide, 2` or `t.add, 0`) must raise "
        "`KeyError`."
    ),
    "stub": (
        "def ex3_back_funcs():\n"
        '    """Return BackFuncs class, registered instance, div_back0, div_back1."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "result = ex3_back_funcs()\n"
        "assert isinstance(result, dict)\n"
        "for k in ('BackFuncs', 'bf', 'div_back0', 'div_back1'):\n"
        "    assert k in result, f'missing key {k}: {list(result.keys())}'\n"
        "\n"
        "BackFuncs = result['BackFuncs']\n"
        "bf = result['bf']\n"
        "div_back0 = result['div_back0']\n"
        "div_back1 = result['div_back1']\n"
        "\n"
        "# === Instance type ===\n"
        "assert isinstance(bf, BackFuncs)\n"
        "\n"
        "# === Lookup by (fn, argnum) returns the right back fn ===\n"
        "fn0 = bf.get_back_func(t.divide, 0)\n"
        "fn1 = bf.get_back_func(t.divide, 1)\n"
        "assert fn0 is div_back0, 'argnum=0 must dispatch to div_back0'\n"
        "assert fn1 is div_back1, 'argnum=1 must dispatch to div_back1'\n"
        "\n"
        "# === Computed gradients match analytic forms ===\n"
        "x = t.tensor([2.0, 4.0, 6.0])\n"
        "y = t.tensor([1.0, 2.0, 3.0])\n"
        "out = x / y\n"
        "grad_out = t.ones_like(out)\n"
        "g0 = fn0(grad_out, out, x, y)\n"
        "g1 = fn1(grad_out, out, x, y)\n"
        "assert t.allclose(g0, grad_out / y), f'div_back0 wrong: {g0}'\n"
        "expected_g1 = -grad_out * out / y\n"
        "assert t.allclose(g1, expected_g1), f'div_back1 wrong: {g1}'\n"
        "\n"
        "# === Missing key raises KeyError ===\n"
        "raised = False\n"
        "try:\n"
        "    bf.get_back_func(t.divide, 2)  # no argnum=2\n"
        "except KeyError:\n"
        "    raised = True\n"
        "assert raised, 'missing (fn, argnum=2) must raise KeyError'\n"
        "\n"
        "raised = False\n"
        "try:\n"
        "    bf.get_back_func(t.add, 0)  # t.add was never registered\n"
        "except KeyError:\n"
        "    raised = True\n"
        "assert raised, 'unregistered fn (t.add) must raise KeyError'\n"
        "\n"
        "# === Registry can be extended in place ===\n"
        "def add_back0(grad_out, out, x, y):\n"
        "    return grad_out\n"
        "bf.add_back_func(t.add, 0, add_back0)\n"
        "assert bf.get_back_func(t.add, 0) is add_back0\n"
        "\n"
        "# === Fresh BackFuncs instance is independent ===\n"
        "bf2 = BackFuncs()\n"
        "raised = False\n"
        "try:\n"
        "    bf2.get_back_func(t.divide, 0)\n"
        "except KeyError:\n"
        "    raised = True\n"
        "assert raised, 'fresh BackFuncs must not share state'\n"
        "\n"
        "# === Cross-check vs torch.autograd on a tracked tensor ===\n"
        "x = t.tensor([3.0, 6.0], requires_grad=True)\n"
        "y = t.tensor([1.5, 2.0], requires_grad=True)\n"
        "out_t = x / y\n"
        "out_t.sum().backward()\n"
        "assert t.allclose(x.grad, fn0(t.ones_like(out_t), out_t.detach(), x.detach(), y.detach()))\n"
        "assert t.allclose(y.grad, fn1(t.ones_like(out_t), out_t.detach(), x.detach(), y.detach()))"
    ),
    "solution_body": (
        "def ex3_back_funcs():\n"
        "    class BackFuncs:\n"
        "        def __init__(self):\n"
        "            self._registry = {}\n"
        "        def add_back_func(self, fwd_fn, argnum, back_fn):\n"
        "            self._registry[(fwd_fn, argnum)] = back_fn\n"
        "        def get_back_func(self, fwd_fn, argnum):\n"
        "            return self._registry[(fwd_fn, argnum)]\n"
        "\n"
        "    def div_back0(grad_out, out, x, y):\n"
        "        return grad_out / y\n"
        "\n"
        "    def div_back1(grad_out, out, x, y):\n"
        "        return -grad_out * out / y\n"
        "\n"
        "    bf = BackFuncs()\n"
        "    bf.add_back_func(t.divide, 0, div_back0)\n"
        "    bf.add_back_func(t.divide, 1, div_back1)\n"
        "    return {'BackFuncs': BackFuncs, 'bf': bf,\n"
        "            'div_back0': div_back0, 'div_back1': div_back1}"
    ),
    "solution_notes": (
        "**Key shape `(fwd_fn, argnum)` is the whole trick.** The forward "
        "fn is the dispatching authority; argnum picks which arg's "
        "gradient. Using anything else (e.g. just the function, or a "
        "string name) breaks composition the moment you register two "
        "back fns for the same fn.\n\n"
        "**Don't catch the KeyError inside `get_back_func`.** Let it "
        "propagate — the reverse pass needs to know if it asks for a "
        "back fn that was never registered. A friendly wrapper that "
        "re-raises with a better message is fine, but swallowing is "
        "actively harmful.\n\n"
        "**Why a class and not just a dict.** The class wrapper is the "
        "extension point — once you add `register_kwarg_back_func`, "
        "`list_registered`, or per-instance defaults, the dict shape "
        "stops being expressive enough."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 2 — chain-rule-elementwise ex3 (leaky_relu_back with slope kwarg)
# ---------------------------------------------------------------------------

SPEC_CHAIN = {
    "atom_id": "chain-rule-elementwise",
    "subtopic": "Backprop: Elementwise chain rule",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_LEAKY_RELU,
    "exercise_index": 3,
    "exercise_title": "leaky_relu_back — parameterized elementwise back fn",
    "slug": "leaky-relu-back-parameterized-elementwise-back-fn",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["leaky-relu", "chain-rule", "kwarg", "parameterized"],
    "kcs": [
        "chain-rule-elementwise",
        "back-fn-accepts-forward-kwargs",
    ],
    "lo": (
        "Apply the elementwise chain rule to a parameterized op by writing "
        "`leaky_relu_back(grad_out, out, x, negative_slope)` so the slope "
        "kwarg threads through and the derivative is `1` where `x > 0`, "
        "`negative_slope` elsewhere."
    ),
    "prompt_body": (
        "Implement `ex3_leaky_relu_back(grad_out, out, x, negative_slope=0.01)`.\n\n"
        "Forward op (for context): `out = leaky_relu(x, negative_slope) = "
        "x if x > 0 else negative_slope * x` (elementwise).\n\n"
        "Local derivative is piecewise:\n"
        "- `1` where `x > 0`\n"
        "- `negative_slope` where `x <= 0`\n\n"
        "Return `grad_out * local_derivative` with shape == `x.shape`.\n\n"
        "Constraints:\n\n"
        "1. The `negative_slope` argument MUST be a keyword arg with default "
        "`0.01` (matching `torch.nn.functional.leaky_relu`).\n"
        "2. Use `t.where(x > 0, ..., ...)` or a `(x > 0).float()` mask — "
        "either is fine, but the output dtype must match `grad_out.dtype`.\n"
        "3. The function must work for any tensor dtype (float32, float64) "
        "and any shape (0-D scalar, 1-D, 2-D, etc.).\n"
        "4. The `out` argument is provided for signature consistency with "
        "ex1/ex2 — you don't have to use it (leaky_relu's derivative depends "
        "on `x`, not `out`)."
    ),
    "stub": (
        "def ex3_leaky_relu_back(grad_out, out, x, negative_slope=0.01):\n"
        '    """Elementwise back fn for leaky_relu — slope threads through as kwarg."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Default slope (0.01) — most common case ===\n"
        "x = t.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])\n"
        "out = t.where(x > 0, x, 0.01 * x)\n"
        "grad_out = t.ones_like(x)\n"
        "g = ex3_leaky_relu_back(grad_out, out, x)\n"
        "expected = t.tensor([0.01, 0.01, 0.01, 1.0, 1.0])\n"
        "# Note: at x == 0, convention is to use the negative-side slope.\n"
        "assert g.shape == x.shape, f'shape mismatch: {g.shape}'\n"
        "assert t.allclose(g, expected), f'default slope: got {g}, expected {expected}'\n"
        "\n"
        "# === Custom slope (0.2 — a typical PReLU choice) ===\n"
        "g = ex3_leaky_relu_back(grad_out, out, x, negative_slope=0.2)\n"
        "expected = t.tensor([0.2, 0.2, 0.2, 1.0, 1.0])\n"
        "assert t.allclose(g, expected), f'slope=0.2: got {g}'\n"
        "\n"
        "# === Slope=0 reduces to relu_back ===\n"
        "g = ex3_leaky_relu_back(grad_out, out, x, negative_slope=0.0)\n"
        "expected_relu = (x > 0).to(grad_out.dtype)\n"
        "assert t.allclose(g, expected_relu), f'slope=0 must equal relu: got {g}'\n"
        "\n"
        "# === Slope=1 reduces to identity (linear pass-through) ===\n"
        "g = ex3_leaky_relu_back(grad_out, out, x, negative_slope=1.0)\n"
        "assert t.allclose(g, grad_out), f'slope=1 must equal grad_out: got {g}'\n"
        "\n"
        "# === Non-unit grad_out — chain-rule scaling ===\n"
        "grad_out2 = t.tensor([10.0, 20.0, 30.0, 40.0, 50.0])\n"
        "g = ex3_leaky_relu_back(grad_out2, out, x, negative_slope=0.01)\n"
        "expected = t.tensor([0.01*10, 0.01*20, 0.01*30, 1.0*40, 1.0*50])\n"
        "assert t.allclose(g, expected), f'scaled grad_out wrong: got {g}'\n"
        "\n"
        "# === 2-D shape ===\n"
        "x = t.randn(4, 5)\n"
        "out = t.where(x > 0, x, 0.01 * x)\n"
        "grad_out = t.randn(4, 5)\n"
        "g = ex3_leaky_relu_back(grad_out, out, x)\n"
        "assert g.shape == (4, 5)\n"
        "# Manually compute expected\n"
        "local = t.where(x > 0, t.ones_like(x), 0.01 * t.ones_like(x))\n"
        "assert t.allclose(g, grad_out * local), 'matrix-shape leaky_relu_back wrong'\n"
        "\n"
        "# === Cross-check vs torch.autograd ===\n"
        "x = t.randn(10, requires_grad=True)\n"
        "out_t = t.nn.functional.leaky_relu(x, negative_slope=0.05)\n"
        "out_t.sum().backward()\n"
        "ours = ex3_leaky_relu_back(t.ones_like(out_t), out_t.detach(), x.detach(), negative_slope=0.05)\n"
        "assert t.allclose(x.grad, ours, atol=1e-6), f'autograd mismatch: {x.grad} vs {ours}'\n"
        "\n"
        "# === Scalar (0-D) input ===\n"
        "x = t.tensor(-3.0)\n"
        "out = t.tensor(-0.03)\n"
        "g = ex3_leaky_relu_back(t.tensor(1.0), out, x, negative_slope=0.01)\n"
        "assert g.dim() == 0\n"
        "assert abs(g.item() - 0.01) < 1e-6, f'0-D negative case: {g.item()}'\n"
        "\n"
        "# === Float64 ===\n"
        "x = t.tensor([-1.0, 1.0], dtype=t.float64)\n"
        "out = t.where(x > 0, x, 0.01 * x)\n"
        "g = ex3_leaky_relu_back(t.ones_like(x), out, x)\n"
        "assert g.dtype == t.float64, f'dtype must match grad_out: got {g.dtype}'"
    ),
    "solution_body": (
        "def ex3_leaky_relu_back(grad_out, out, x, negative_slope=0.01):\n"
        "    local = t.where(\n"
        "        x > 0,\n"
        "        t.ones_like(x),\n"
        "        t.full_like(x, negative_slope),\n"
        "    )\n"
        "    return grad_out * local"
    ),
    "solution_notes": (
        "**`t.where(cond, true_val, false_val)` over `*`.** Multiplying "
        "by `(x > 0).float() + negative_slope * (x <= 0).float()` works "
        "but allocates two masks. `t.where` does it in one pass.\n\n"
        "**`t.full_like(x, negative_slope)` matches dtype/device of `x`.** "
        "Using a bare Python float on the false-branch would force PyTorch "
        "to cast, sometimes silently upgrading to float64 and breaking "
        "downstream dtype expectations.\n\n"
        "**Convention at `x == 0`.** Mathematically the derivative is "
        "undefined exactly at the kink. PyTorch and most frameworks use "
        "the negative-side slope. Our `x > 0` (strict) condition matches "
        "that convention — at zero, we fall into the false branch."
    ),
    "extra_imports": ["import torch.nn.functional as F"],
}


# ---------------------------------------------------------------------------
# SPEC 3 — grad-tracking-global-toggle ex3 (set_grad_enabled value setter)
# ---------------------------------------------------------------------------

SPEC_TOGGLE = {
    "atom_id": "grad-tracking-global-toggle",
    "subtopic": "Backprop: Grad-tracking toggle",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_SET_GRAD_ENABLED,
    "exercise_index": 3,
    "exercise_title": "set_grad_enabled(mode) — value-taking context manager",
    "slug": "set-grad-enabled-value-taking-context-manager",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["set_grad_enabled", "context-manager", "value-setter", "nesting"],
    "kcs": [
        "grad-tracking-global-toggle",
        "context-manager-restores-snapshot",
    ],
    "lo": (
        "Apply the snapshot-and-restore context-manager pattern to a setter "
        "that takes an EXPLICIT bool value (not just 'force off'), so nested "
        "set_grad_enabled(True) can re-enable grad inside an outer no-grad "
        "block."
    ),
    "prompt_body": (
        "Implement `ex3_set_grad_enabled()` returning a class "
        "`SetGradEnabled` and a 'getter' fn `get_state()`.\n\n"
        "The class must:\n\n"
        "1. `__init__(self, mode: bool)` — store `mode` as `self.mode`.\n"
        "2. `__enter__(self)` — snapshot the CURRENT module-level "
        "`grad_tracking_enabled` into `self.prev`, then set the module "
        "global to `self.mode`. Return `self`.\n"
        "3. `__exit__(self, exc_type, exc_val, tb)` — restore the snapshot. "
        "Return `False` so exceptions propagate.\n\n"
        "Module state:\n\n"
        "- Define `grad_tracking_enabled = True` at module/global scope "
        "BEFORE the class.\n"
        "- The class MUST read/write the GLOBAL via `globals()` or "
        "explicit `global grad_tracking_enabled` — never snapshot it into "
        "the closure.\n\n"
        "Helper:\n\n"
        "- `get_state()` returns the current `grad_tracking_enabled` (so "
        "the test can poll it without depending on the test's own scope).\n\n"
        "Return: `{'SetGradEnabled': SetGradEnabled, 'get_state': get_state, "
        "'reset': reset}` where `reset()` sets the global back to `True`."
    ),
    "stub": (
        "def ex3_set_grad_enabled():\n"
        '    """Return {SetGradEnabled class, get_state(), reset()} with module-global toggle."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "result = ex3_set_grad_enabled()\n"
        "SetGradEnabled = result['SetGradEnabled']\n"
        "get_state = result['get_state']\n"
        "reset = result['reset']\n"
        "\n"
        "# === Reset baseline ===\n"
        "reset()\n"
        "assert get_state() is True, 'reset must set toggle to True'\n"
        "\n"
        "# === Force off ===\n"
        "with SetGradEnabled(False):\n"
        "    assert get_state() is False, 'inside False block toggle must be False'\n"
        "assert get_state() is True, 'exit must restore prior True'\n"
        "\n"
        "# === Force on explicitly (no-op when global is already True) ===\n"
        "with SetGradEnabled(True):\n"
        "    assert get_state() is True\n"
        "assert get_state() is True\n"
        "\n"
        "# === The headline test: re-enable inside a no-grad block ===\n"
        "reset()\n"
        "with SetGradEnabled(False):                  # outer: off\n"
        "    assert get_state() is False\n"
        "    with SetGradEnabled(True):               # inner: re-enable\n"
        "        assert get_state() is True, 'inner True must override outer False'\n"
        "    assert get_state() is False, 'after inner exit, outer False restored'\n"
        "assert get_state() is True, 'after outer exit, original True restored'\n"
        "\n"
        "# === Deep nesting (3 levels) ===\n"
        "reset()\n"
        "with SetGradEnabled(False):\n"
        "    with SetGradEnabled(True):\n"
        "        with SetGradEnabled(False):\n"
        "            assert get_state() is False\n"
        "        assert get_state() is True, 'level 3 exit -> level 2'\n"
        "    assert get_state() is False, 'level 2 exit -> level 1'\n"
        "assert get_state() is True, 'level 1 exit -> level 0'\n"
        "\n"
        "# === Restore on exception ===\n"
        "reset()\n"
        "raised = False\n"
        "try:\n"
        "    with SetGradEnabled(False):\n"
        "        raise RuntimeError('boom')\n"
        "except RuntimeError:\n"
        "    raised = True\n"
        "assert raised\n"
        "assert get_state() is True, 'exception must NOT prevent restoration'\n"
        "\n"
        "# === self.prev is captured per-instance (instances don't share state) ===\n"
        "reset()\n"
        "ctx_a = SetGradEnabled(False)\n"
        "ctx_b = SetGradEnabled(False)\n"
        "with ctx_a:\n"
        "    with ctx_b:\n"
        "        assert get_state() is False\n"
        "    assert get_state() is False, 'ctx_b restores to its own prev (False)'\n"
        "assert get_state() is True, 'ctx_a restores to original True'\n"
        "\n"
        "# === __enter__ returns self (so `as` binding works) ===\n"
        "reset()\n"
        "with SetGradEnabled(False) as ctx:\n"
        "    assert ctx is not None\n"
        "    assert isinstance(ctx, SetGradEnabled)\n"
        "\n"
        "reset()"
    ),
    "solution_body": (
        "def ex3_set_grad_enabled():\n"
        "    # Module-level state lives in a dict so the closure can mutate it.\n"
        "    state = {'grad_tracking_enabled': True}\n"
        "\n"
        "    class SetGradEnabled:\n"
        "        def __init__(self, mode):\n"
        "            self.mode = bool(mode)\n"
        "            self.prev = None\n"
        "        def __enter__(self):\n"
        "            self.prev = state['grad_tracking_enabled']\n"
        "            state['grad_tracking_enabled'] = self.mode\n"
        "            return self\n"
        "        def __exit__(self, exc_type, exc_val, tb):\n"
        "            state['grad_tracking_enabled'] = self.prev\n"
        "            return False\n"
        "\n"
        "    def get_state():\n"
        "        return state['grad_tracking_enabled']\n"
        "\n"
        "    def reset():\n"
        "        state['grad_tracking_enabled'] = True\n"
        "\n"
        "    return {'SetGradEnabled': SetGradEnabled, 'get_state': get_state,\n"
        "            'reset': reset}"
    ),
    "solution_notes": (
        "**Per-instance `self.prev` is the snapshot key.** Each context "
        "instance captures the value AT THE MOMENT OF `__enter__` — not "
        "at construction. That's why you can write `ctx_a = SetGradEnabled"
        "(False)` and only have it take effect inside `with ctx_a:`.\n\n"
        "**Why a state dict instead of a real module global.** Python's "
        "`global` declarations work inside a real module but not inside a "
        "function closure. Wrapping the toggle in a dict gives us a single "
        "mutable cell that all inner functions can share without "
        "import-time gymnastics. The behaviour is identical from the "
        "outside.\n\n"
        "**`return False` from `__exit__` re-raises exceptions.** "
        "Returning truthy would suppress the exception — wrong for a "
        "grad-tracking utility, which has no business swallowing errors."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 4 — kwargs-pass-through-recipe ex3 (replay via **recipe.kwargs)
# ---------------------------------------------------------------------------

SPEC_KWARGS = {
    "atom_id": "kwargs-pass-through-recipe",
    "subtopic": "Backprop: Kwargs pass-through",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_RECIPE_KWARG_REPLAY,
    "exercise_index": 3,
    "exercise_title": "replay back fn via **recipe.kwargs — recover dim at reverse time",
    "slug": "replay-back-fn-via-recipe-kwargs-recover-dim",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["recipe", "kwargs", "replay", "sum-back", "dim"],
    "kcs": [
        "kwargs-pass-through-recipe",
        "back-fn-replays-via-splat",
    ],
    "lo": (
        "Apply the `**recipe.kwargs` splat pattern to invoke a back fn "
        "(sum_back) using only the Recipe — recover the `dim` and "
        "`keepdim` the forward used without storing them anywhere else."
    ),
    "prompt_body": (
        "We give you a `Recipe` dataclass (4 fields: `func`, `args`, "
        "`kwargs`, `parents`) and a `sum_back` implementation. Your job is "
        "to implement `ex3_replay_sum_back(out_tensor, grad_out)`.\n\n"
        "Inputs:\n"
        "- `out_tensor`: the OUTPUT of a forward `sum`. It carries a "
        "`.array` field (the raw torch tensor) and a `.recipe` field (the "
        "Recipe stored at forward time). The Recipe's `kwargs` will be "
        "something like `{'dim': 1, 'keepdim': True}` or `{}`.\n"
        "- `grad_out`: a raw torch tensor with the same shape as "
        "`out_tensor.array`.\n\n"
        "Steps:\n\n"
        "1. Recover the parent's raw tensor: `parent_arr = "
        "out_tensor.recipe.parents[0].array`.\n"
        "2. Call `sum_back(grad_out, out_tensor.array, parent_arr, "
        "**out_tensor.recipe.kwargs)`.\n"
        "3. Return the resulting `grad_in` tensor.\n\n"
        "The CRITICAL constraint: you must use `**recipe.kwargs` to splat "
        "the stored kwargs into the back-fn call — NOT manually unpack or "
        "hard-code `dim` / `keepdim`. The whole point is the recipe-"
        "replay contract."
    ),
    "stub": (
        "def ex3_replay_sum_back(out_tensor, grad_out):\n"
        '    """Invoke sum_back using **recipe.kwargs splat to recover dim/keepdim."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "from dataclasses import dataclass\n"
        "from typing import Callable\n"
        "\n"
        "@dataclass\n"
        "class Recipe:\n"
        "    func: Callable\n"
        "    args: tuple\n"
        "    kwargs: dict\n"
        "    parents: dict\n"
        "\n"
        "class MiniTensor:\n"
        "    def __init__(self, array, recipe=None):\n"
        "        self.array = array\n"
        "        self.recipe = recipe\n"
        "\n"
        "def sum_back(grad_out, out, x, **kwargs):\n"
        "    # Reverse: re-expand grad_out along the reduced dim(s) back to x.shape.\n"
        "    dim = kwargs.get('dim', None)\n"
        "    keepdim = kwargs.get('keepdim', False)\n"
        "    if dim is None:\n"
        "        return grad_out * t.ones_like(x)\n"
        "    if not keepdim:\n"
        "        grad_out = grad_out.unsqueeze(dim)\n"
        "    return grad_out.expand_as(x)\n"
        "\n"
        "# Expose sum_back to the function via __globals__ injection (test scope).\n"
        "# In real ARENA code, sum_back is module-level — we mimic that here.\n"
        "ex3_replay_sum_back.__globals__['sum_back'] = sum_back\n"
        "\n"
        "# === Case A: dim=1, keepdim=False ===\n"
        "x = t.arange(12.0).reshape(3, 4)\n"
        "kwargs = {'dim': 1, 'keepdim': False}\n"
        "out_arr = x.sum(**kwargs)  # shape (3,)\n"
        "parent = MiniTensor(x)\n"
        "out_tensor = MiniTensor(out_arr, Recipe(t.sum, (x,), kwargs, {0: parent}))\n"
        "grad_out = t.ones_like(out_arr)\n"
        "grad_in = ex3_replay_sum_back(out_tensor, grad_out)\n"
        "expected = t.ones_like(x)  # d(sum)/dx_ij = 1 for each element in the reduced row\n"
        "assert grad_in.shape == x.shape, f'shape mismatch: {grad_in.shape}'\n"
        "assert t.allclose(grad_in, expected), f'dim=1: {grad_in}'\n"
        "\n"
        "# === Case B: dim=0, keepdim=True ===\n"
        "x = t.arange(20.0).reshape(4, 5)\n"
        "kwargs = {'dim': 0, 'keepdim': True}\n"
        "out_arr = x.sum(**kwargs)  # shape (1, 5)\n"
        "parent = MiniTensor(x)\n"
        "out_tensor = MiniTensor(out_arr, Recipe(t.sum, (x,), kwargs, {0: parent}))\n"
        "grad_out = t.ones_like(out_arr)\n"
        "grad_in = ex3_replay_sum_back(out_tensor, grad_out)\n"
        "expected = t.ones_like(x)\n"
        "assert grad_in.shape == x.shape\n"
        "assert t.allclose(grad_in, expected)\n"
        "\n"
        "# === Case C: empty kwargs (reduce-all) ===\n"
        "x = t.arange(6.0).reshape(2, 3)\n"
        "kwargs = {}\n"
        "out_arr = x.sum()  # scalar\n"
        "parent = MiniTensor(x)\n"
        "out_tensor = MiniTensor(out_arr, Recipe(t.sum, (x,), kwargs, {0: parent}))\n"
        "grad_out = t.tensor(2.0)\n"
        "grad_in = ex3_replay_sum_back(out_tensor, grad_out)\n"
        "assert grad_in.shape == x.shape\n"
        "expected = 2.0 * t.ones_like(x)\n"
        "assert t.allclose(grad_in, expected), f'reduce-all: {grad_in}'\n"
        "\n"
        "# === Case D: non-unit grad_out scales linearly ===\n"
        "x = t.arange(12.0).reshape(3, 4)\n"
        "kwargs = {'dim': 1, 'keepdim': False}\n"
        "out_arr = x.sum(**kwargs)\n"
        "parent = MiniTensor(x)\n"
        "out_tensor = MiniTensor(out_arr, Recipe(t.sum, (x,), kwargs, {0: parent}))\n"
        "grad_out = t.tensor([1.0, 2.0, 3.0])\n"
        "grad_in = ex3_replay_sum_back(out_tensor, grad_out)\n"
        "expected = t.tensor([[1.]*4, [2.]*4, [3.]*4])\n"
        "assert t.allclose(grad_in, expected), f'scaled grad_out: {grad_in}'\n"
        "\n"
        "# === Case E: cross-check vs torch.autograd ===\n"
        "x = t.randn(3, 4, requires_grad=True)\n"
        "y = x.sum(dim=1, keepdim=False)\n"
        "y.sum().backward()\n"
        "parent = MiniTensor(x.detach())\n"
        "out_tensor = MiniTensor(y.detach(),\n"
        "                       Recipe(t.sum, (x.detach(),), {'dim': 1, 'keepdim': False}, {0: parent}))\n"
        "grad_in = ex3_replay_sum_back(out_tensor, t.ones_like(y))\n"
        "assert t.allclose(x.grad, grad_in, atol=1e-6), f'autograd mismatch'"
    ),
    "solution_body": (
        "def ex3_replay_sum_back(out_tensor, grad_out):\n"
        "    recipe = out_tensor.recipe\n"
        "    parent_arr = recipe.parents[0].array\n"
        "    # Splat the stored kwargs into the back fn — dim, keepdim, etc.\n"
        "    return sum_back(grad_out, out_tensor.array, parent_arr, **recipe.kwargs)"
    ),
    "solution_notes": (
        "**`**recipe.kwargs` IS the contract.** Without the splat you'd "
        "either hard-code keys (`recipe.kwargs.get('dim')`) — coupling "
        "the dispatcher to every op's signature — or write per-op "
        "dispatchers. Splatting makes one dispatcher work for `sum_back`, "
        "`mean_back`, `softmax_back`, etc., as long as each back fn's "
        "signature mirrors its forward's.\n\n"
        "**Empty kwargs `**{}` is still valid splat syntax.** Python "
        "accepts `f(**{})` as 'call with no extra kwargs'. So ex2's "
        "empty-kwargs invariant is what makes this drill's replay work "
        "uniformly across reduce-all and reduce-along-dim cases.\n\n"
        "**`recipe.parents[0]` for unary ops.** Multi-input ops "
        "iterate `recipe.parents.items()` and call the appropriate "
        "argnum-keyed back fn for each — that's the next drill (parents-"
        "dispatch). Here we focus on the kwargs-replay piece in "
        "isolation."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 5 — parents-dict-by-argidx ex3 (dispatch back fns over parents.items())
# ---------------------------------------------------------------------------

SPEC_PARENTS = {
    "atom_id": "parents-dict-by-argidx",
    "subtopic": "Backprop: Parents dict by argidx",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_PARENTS_DISPATCH,
    "exercise_index": 3,
    "exercise_title": "dispatch back fns over parents.items() — int-keyed positional only",
    "slug": "dispatch-back-fns-over-parents-items-int-keyed",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["parents", "dispatch", "argidx", "iter-items"],
    "kcs": [
        "parents-dict-by-argidx",
        "back-fn-lookup-by-fn-argnum",
    ],
    "lo": (
        "Apply the parents-dict dispatch pattern: iterate "
        "`recipe.parents.items()` filtering to int keys, look up each "
        "back fn by `(recipe.func, argnum)`, and return a mapping from "
        "argnum to the computed grad tensor."
    ),
    "prompt_body": (
        "Implement `ex3_dispatch_back_fns(recipe, grad_out, out_arr, "
        "back_funcs)`. Iterate over `recipe.parents.items()`, skip "
        "non-int keys (they're kwarg-name keys for kwarg-positioned "
        "Tensors), and for each `(argnum, parent_tensor)`:\n\n"
        "1. Look up the back fn: `back_fn = back_funcs[(recipe.func, "
        "argnum)]`. If missing, raise `KeyError` (let the natural error "
        "propagate).\n"
        "2. Call `back_fn(grad_out, out_arr, *recipe.args, "
        "**recipe.kwargs)` — splat both stored args and kwargs.\n"
        "3. Collect the result keyed by `argnum`.\n\n"
        "Return a `dict[int, Tensor]` mapping argnum → computed grad.\n\n"
        "Inputs you can rely on:\n"
        "- `recipe.func`: the forward fn (e.g. `t.add`).\n"
        "- `recipe.args`: tuple of raw (unboxed) args used in the forward.\n"
        "- `recipe.kwargs`: dict (possibly empty).\n"
        "- `recipe.parents`: dict mixing int and str keys.\n"
        "- `back_funcs`: dict keyed by `(fwd_fn, argnum: int)`.\n\n"
        "Constraints:\n"
        "- DO NOT compute grads for str-keyed parents (kwarg Tensors). "
        "Filter them out via `isinstance(k, int)`.\n"
        "- DO NOT mutate `recipe`."
    ),
    "stub": (
        "def ex3_dispatch_back_fns(recipe, grad_out, out_arr, back_funcs):\n"
        '    """Iterate parents.items(), dispatch back fn per argnum, return {argnum: grad}."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "from dataclasses import dataclass\n"
        "from typing import Callable\n"
        "\n"
        "@dataclass\n"
        "class Recipe:\n"
        "    func: Callable\n"
        "    args: tuple\n"
        "    kwargs: dict\n"
        "    parents: dict\n"
        "\n"
        "class MiniTensor:\n"
        "    def __init__(self, array):\n"
        "        self.array = array\n"
        "\n"
        "# === Binary add: 2 positional parents ===\n"
        "def add_back0(grad_out, out, x, y, **_): return grad_out * t.ones_like(x)\n"
        "def add_back1(grad_out, out, x, y, **_): return grad_out * t.ones_like(y)\n"
        "back_funcs = {(t.add, 0): add_back0, (t.add, 1): add_back1}\n"
        "\n"
        "x = MiniTensor(t.tensor([1.0, 2.0]))\n"
        "y = MiniTensor(t.tensor([3.0, 4.0]))\n"
        "out_arr = x.array + y.array\n"
        "recipe = Recipe(t.add, (x.array, y.array), {}, {0: x, 1: y})\n"
        "grad_out = t.ones_like(out_arr)\n"
        "\n"
        "result = ex3_dispatch_back_fns(recipe, grad_out, out_arr, back_funcs)\n"
        "assert isinstance(result, dict)\n"
        "assert set(result.keys()) == {0, 1}, f'expected argnums 0,1; got {list(result.keys())}'\n"
        "assert t.allclose(result[0], t.ones_like(x.array))\n"
        "assert t.allclose(result[1], t.ones_like(y.array))\n"
        "\n"
        "# === Unary log: 1 parent ===\n"
        "def log_back0(grad_out, out, x, **_): return grad_out / x\n"
        "back_funcs = {(t.log, 0): log_back0}\n"
        "x = MiniTensor(t.tensor([1.0, 2.0, 4.0]))\n"
        "out_arr = t.log(x.array)\n"
        "recipe = Recipe(t.log, (x.array,), {}, {0: x})\n"
        "grad_out = t.ones_like(out_arr)\n"
        "result = ex3_dispatch_back_fns(recipe, grad_out, out_arr, back_funcs)\n"
        "assert set(result.keys()) == {0}\n"
        "assert t.allclose(result[0], 1.0 / x.array)\n"
        "\n"
        "# === Skip str-keyed parents (kwarg Tensors) ===\n"
        "def add_back0_skip(grad_out, out, x, **_): return grad_out * t.ones_like(x)\n"
        "back_funcs = {(t.add, 0): add_back0_skip}\n"
        "x = MiniTensor(t.tensor([1.0]))\n"
        "kwarg_t = MiniTensor(t.tensor([99.0]))  # str-keyed parent — must be skipped\n"
        "out_arr = t.tensor([1.0])\n"
        "recipe = Recipe(t.add, (x.array,), {'mask': kwarg_t.array}, {0: x, 'mask': kwarg_t})\n"
        "result = ex3_dispatch_back_fns(recipe, t.ones_like(out_arr), out_arr, back_funcs)\n"
        "assert set(result.keys()) == {0}, f'must skip str-keyed; got {list(result.keys())}'\n"
        "assert 'mask' not in result\n"
        "\n"
        "# === Empty parents (no Tensor inputs — degenerate) ===\n"
        "recipe = Recipe(t.add, (), {}, {})\n"
        "result = ex3_dispatch_back_fns(recipe, t.tensor(1.0), t.tensor(0.0), {})\n"
        "assert result == {}, f'empty parents must yield empty dict; got {result}'\n"
        "\n"
        "# === Missing back fn raises KeyError ===\n"
        "x = MiniTensor(t.tensor([1.0]))\n"
        "recipe = Recipe(t.add, (x.array,), {}, {0: x})  # no entry registered\n"
        "raised = False\n"
        "try:\n"
        "    ex3_dispatch_back_fns(recipe, t.ones_like(x.array), x.array, {})\n"
        "except KeyError:\n"
        "    raised = True\n"
        "assert raised, 'missing back fn must raise KeyError'\n"
        "\n"
        "# === Kwargs splat to back fn ===\n"
        "# Define a back fn that DEPENDS on a kwarg to verify the splat reaches it.\n"
        "def sum_back0(grad_out, out, x, dim=None, **_):\n"
        "    if dim is None:\n"
        "        return grad_out * t.ones_like(x)\n"
        "    return grad_out.unsqueeze(dim).expand_as(x)\n"
        "back_funcs = {(t.sum, 0): sum_back0}\n"
        "x = MiniTensor(t.arange(6.0).reshape(2, 3))\n"
        "out_arr = x.array.sum(dim=1)\n"
        "recipe = Recipe(t.sum, (x.array,), {'dim': 1}, {0: x})\n"
        "grad_out = t.tensor([1.0, 2.0])\n"
        "result = ex3_dispatch_back_fns(recipe, grad_out, out_arr, back_funcs)\n"
        "expected = t.tensor([[1., 1., 1.], [2., 2., 2.]])\n"
        "assert t.allclose(result[0], expected), f'kwargs splat broken: {result[0]}'\n"
        "\n"
        "# === Recipe is not mutated ===\n"
        "x = MiniTensor(t.tensor([1.0]))\n"
        "back_funcs = {(t.log, 0): lambda g, o, x, **_: g / x}\n"
        "recipe = Recipe(t.log, (x.array,), {}, {0: x})\n"
        "before_parents = dict(recipe.parents)\n"
        "before_kwargs = dict(recipe.kwargs)\n"
        "_ = ex3_dispatch_back_fns(recipe, t.ones(1), t.zeros(1), back_funcs)\n"
        "assert recipe.parents == before_parents, 'parents was mutated'\n"
        "assert recipe.kwargs == before_kwargs, 'kwargs was mutated'"
    ),
    "solution_body": (
        "def ex3_dispatch_back_fns(recipe, grad_out, out_arr, back_funcs):\n"
        "    grads = {}\n"
        "    for argnum, parent in recipe.parents.items():\n"
        "        if not isinstance(argnum, int):\n"
        "            continue  # kwarg-named parents handled by a different dispatcher\n"
        "        back_fn = back_funcs[(recipe.func, argnum)]\n"
        "        grads[argnum] = back_fn(\n"
        "            grad_out, out_arr, *recipe.args, **recipe.kwargs\n"
        "        )\n"
        "    return grads"
    ),
    "solution_notes": (
        "**`isinstance(argnum, int)` filter is critical.** ex2 mixed int "
        "and str keys in parents. The positional back-fn dispatcher must "
        "skip the str keys — they belong to a parallel kwarg-back-fn "
        "table that this drill doesn't model. Without the filter, you'd "
        "do `back_funcs[(recipe.func, 'mask')]` which crashes.\n\n"
        "**`*recipe.args, **recipe.kwargs` is the universal call shape.** "
        "Every back fn's signature is `(grad_out, out, *forward_args, "
        "**forward_kwargs)`. Splatting works for unary, binary, ternary, "
        "and parameterized ops uniformly — that's the whole point of the "
        "Recipe abstraction.\n\n"
        "**The dispatcher is intentionally minimal.** No grad accumulation "
        "(that's the leaf-accumulator's job); no graph traversal (that's "
        "topological sort's job). Just: 'given a Recipe and an incoming "
        "grad, hand back the per-argnum grad contributions'. Single "
        "responsibility."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 6 — recipe-dataclass ex3 (clamp_forward: 1 parent + 2 kwargs)
# ---------------------------------------------------------------------------

SPEC_RECIPE = {
    "atom_id": "recipe-dataclass",
    "subtopic": "Backprop: Recipe dataclass",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_CLAMP_FORWARD,
    "exercise_index": 3,
    "exercise_title": "clamp_forward — Recipe with 1 parent and 2 non-empty kwargs",
    "slug": "clamp-forward-recipe-one-parent-two-kwargs",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["recipe", "clamp", "kwargs-non-empty", "scalar-kwarg"],
    "kcs": [
        "recipe-dataclass",
        "recipe-kwargs-stores-non-tensor-args",
    ],
    "lo": (
        "Apply the 4-field Recipe construction to `clamp_forward(x, min, "
        "max)` where the Recipe has ONE Tensor parent and TWO non-Tensor "
        "kwargs (min, max) — distinguishing what goes into parents vs "
        "what goes into kwargs."
    ),
    "prompt_body": (
        "Implement `ex3_clamp_forward(x, *, min_val, max_val)` for "
        "`MiniTensor` input `x` (the `min_val` / `max_val` naming avoids "
        "shadowing Python builtins inside the implementation).\n\n"
        "Required behaviour:\n\n"
        "1. Compute `out_raw = x.array.clamp(min=min_val, max=max_val)`.\n"
        "2. Return a NEW `MiniTensor` with `.array = out_raw` and a fully "
        "populated `.recipe` such that:\n"
        "   - `recipe.func` is `t.clamp` (the forward op).\n"
        "   - `recipe.args` is `(x.array,)` — one-tuple of the unboxed "
        "Tensor input.\n"
        "   - `recipe.kwargs` is `{'min': min_val, 'max': max_val}` — "
        "EXACT keys `'min'` and `'max'` (not `'min_val'`), and the SAME "
        "numeric values that were passed in.\n"
        "   - `recipe.parents` is `{0: x}` — ONLY the positional Tensor "
        "parent. The `min_val` and `max_val` floats DO NOT go into "
        "parents (non-Tensor → no grad flow).\n"
        "\n"
        "The MiniTensor class and Recipe dataclass are pre-defined in the "
        "test cell — you just construct and return.\n\n"
        "Why the kwargs naming flip matters: at backward time, "
        "`clamp_back(grad_out, out, x, **recipe.kwargs)` must receive "
        "`min=..., max=...` (matching `torch.clamp`'s own signature). "
        "Storing under `'min_val'`/`'max_val'` would break the splat."
    ),
    "stub": (
        "def ex3_clamp_forward(x, *, min_val, max_val):\n"
        '    """Forward op for clamp — returns MiniTensor with fully populated Recipe."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "from dataclasses import dataclass, field\n"
        "from typing import Callable\n"
        "\n"
        "@dataclass\n"
        "class Recipe:\n"
        "    func: Callable\n"
        "    args: tuple\n"
        "    kwargs: dict\n"
        "    parents: dict\n"
        "\n"
        "class MiniTensor:\n"
        "    def __init__(self, array, recipe=None):\n"
        "        self.array = array\n"
        "        self.recipe = recipe\n"
        "\n"
        "# Expose MiniTensor / Recipe to the solution via its __globals__.\n"
        "ex3_clamp_forward.__globals__['MiniTensor'] = MiniTensor\n"
        "ex3_clamp_forward.__globals__['Recipe'] = Recipe\n"
        "\n"
        "# === Basic correctness ===\n"
        "x = MiniTensor(t.tensor([-2.0, -0.5, 0.5, 1.5, 3.0]))\n"
        "out = ex3_clamp_forward(x, min_val=-1.0, max_val=1.0)\n"
        "assert isinstance(out, MiniTensor)\n"
        "expected = t.tensor([-1.0, -0.5, 0.5, 1.0, 1.0])\n"
        "assert t.allclose(out.array, expected), f'clamp values wrong: {out.array}'\n"
        "\n"
        "# === Recipe shape ===\n"
        "r = out.recipe\n"
        "assert r is not None, 'recipe must be attached'\n"
        "assert isinstance(r, Recipe)\n"
        "assert r.func is t.clamp, f'recipe.func must be t.clamp; got {r.func}'\n"
        "\n"
        "# === args is the unboxed positional ===\n"
        "assert isinstance(r.args, tuple)\n"
        "assert len(r.args) == 1, f'expected 1-tuple of args; got {r.args}'\n"
        "assert r.args[0] is x.array, 'recipe.args[0] must be the SAME tensor object as x.array (unboxed)'\n"
        "\n"
        "# === kwargs has min/max keys (NOT min_val/max_val) ===\n"
        "assert isinstance(r.kwargs, dict)\n"
        "assert set(r.kwargs.keys()) == {'min', 'max'}, f'wrong kwargs keys: {list(r.kwargs.keys())}'\n"
        "assert r.kwargs['min'] == -1.0\n"
        "assert r.kwargs['max'] == 1.0\n"
        "\n"
        "# === parents has ONLY position 0 — kwargs floats do NOT appear ===\n"
        "assert isinstance(r.parents, dict)\n"
        "assert set(r.parents.keys()) == {0}, f'parents must be {{0}}, got {set(r.parents.keys())}'\n"
        "assert r.parents[0] is x, 'parents[0] must be the boxed MiniTensor, not unboxed array'\n"
        "\n"
        "# === Different min/max values ===\n"
        "x2 = MiniTensor(t.tensor([0.0, 5.0, 10.0]))\n"
        "out2 = ex3_clamp_forward(x2, min_val=3.0, max_val=7.0)\n"
        "assert t.allclose(out2.array, t.tensor([3.0, 5.0, 7.0]))\n"
        "assert out2.recipe.kwargs == {'min': 3.0, 'max': 7.0}\n"
        "\n"
        "# === Splat back into torch.clamp via **recipe.kwargs works ===\n"
        "x3 = MiniTensor(t.tensor([-5.0, 0.0, 5.0]))\n"
        "out3 = ex3_clamp_forward(x3, min_val=-2.0, max_val=2.0)\n"
        "# Replay via splat — same shape, same result.\n"
        "replay = t.clamp(out3.recipe.args[0], **out3.recipe.kwargs)\n"
        "assert t.allclose(replay, out3.array), 'splat replay must produce identical result'\n"
        "\n"
        "# === Output is a NEW MiniTensor (not x mutated) ===\n"
        "x4 = MiniTensor(t.tensor([10.0, -10.0]))\n"
        "out4 = ex3_clamp_forward(x4, min_val=-1.0, max_val=1.0)\n"
        "assert out4 is not x4, 'must return a new MiniTensor, not the input'\n"
        "assert t.allclose(x4.array, t.tensor([10.0, -10.0])), 'input must not be mutated'\n"
        "\n"
        "# === 2-D input ===\n"
        "x5 = MiniTensor(t.tensor([[-3.0, 0.0, 3.0], [-1.5, 0.5, 1.5]]))\n"
        "out5 = ex3_clamp_forward(x5, min_val=-1.0, max_val=1.0)\n"
        "assert out5.array.shape == (2, 3)\n"
        "assert t.allclose(out5.array, t.tensor([[-1.0, 0.0, 1.0], [-1.0, 0.5, 1.0]]))\n"
        "assert out5.recipe.parents == {0: x5}"
    ),
    "solution_body": (
        "def ex3_clamp_forward(x, *, min_val, max_val):\n"
        "    out_raw = x.array.clamp(min=min_val, max=max_val)\n"
        "    recipe = Recipe(\n"
        "        func=t.clamp,\n"
        "        args=(x.array,),\n"
        "        kwargs={'min': min_val, 'max': max_val},\n"
        "        parents={0: x},\n"
        "    )\n"
        "    out = MiniTensor(out_raw, recipe=recipe)\n"
        "    return out"
    ),
    "solution_notes": (
        "**`'min'` not `'min_val'` for kwargs keys.** The naming is for "
        "the Python function signature (avoiding the `min` builtin); the "
        "stored keys must match what `torch.clamp` accepts as kwargs. "
        "Splat at backward time: `t.clamp(..., **recipe.kwargs)` would "
        "fail with 'unexpected keyword argument min_val' otherwise.\n\n"
        "**Floats stay in kwargs, never in parents.** `min_val` and "
        "`max_val` are scalars — `isinstance(x, MiniTensor)` is False — "
        "so they're filtered out of parents (same rule as ex1's parents "
        "drill). They appear ONLY in `recipe.kwargs` because the back fn "
        "needs them to know where the clipping happened.\n\n"
        "**`parents[0] is x` — boxed, not unboxed.** `recipe.args[0]` is "
        "`x.array` (unboxed for the forward call). `recipe.parents[0]` "
        "is the original boxed `MiniTensor` (so the reverse pass can "
        "walk back to its own recipe and continue the chain). This dual "
        "storage is intentional."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 7 — requires-grad-propagation ex3 (chain composition through 2 ops)
# ---------------------------------------------------------------------------

SPEC_REQGRAD = {
    "atom_id": "requires-grad-propagation",
    "subtopic": "Backprop: requires_grad propagation",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_REQGRAD_CHAIN,
    "exercise_index": 3,
    "exercise_title": "chained ops — requires_grad flows through composition",
    "slug": "chained-ops-requires-grad-flows-through-composition",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["requires-grad", "composition", "graph", "propagation"],
    "kcs": [
        "requires-grad-propagation",
        "graph-propagation-is-transitive",
    ],
    "lo": (
        "Analyze how `requires_grad` propagates through a chain of ops: "
        "applying the three-gate rule at each forward op produces a "
        "tensor whose `requires_grad` flag is True iff at least one "
        "original input had it AND every op in the chain was "
        "differentiable."
    ),
    "prompt_body": (
        "Implement `ex3_chain_propagate(inputs, op_table)`. Simulates a "
        "chain of forward ops and reports the final tensor's "
        "`requires_grad`.\n\n"
        "Inputs:\n"
        "- `inputs`: list of `MiniTensor` objects with `.requires_grad` "
        "attribute (bool). These are the LEAF inputs at the bottom of "
        "the chain.\n"
        "- `op_table`: list of dicts, each describing one op to apply. "
        "Each dict has:\n"
        "  - `'op_inputs'`: list of indices into the previous step's "
        "tensors (where the first step indexes into `inputs`). At each "
        "step, those tensors become this op's inputs.\n"
        "  - `'is_differentiable'`: bool — the op's differentiability "
        "flag.\n"
        "  - `'grad_tracking_enabled'`: bool — the global toggle as of "
        "this op.\n\n"
        "Behaviour at each step:\n"
        "- Gather the step's input MiniTensors using `op_inputs` indices.\n"
        "- Compute the OUTPUT MiniTensor's `requires_grad` via the "
        "three-gate AND:\n"
        "  `grad_tracking_enabled AND is_differentiable AND any(input.requires_grad for input in step_inputs)`.\n"
        "- The output becomes the SINGLE tensor available to the next "
        "step at index 0. (For chain simulation we collapse to one output "
        "per step.)\n\n"
        "Return: `bool` — the final step's output `requires_grad`.\n\n"
        "Edge case: if `op_table` is empty, return `False` (no ops, no "
        "output)."
    ),
    "stub": (
        "def ex3_chain_propagate(inputs, op_table):\n"
        '    """Simulate forward chain. Return final tensor requires_grad bool."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "class MiniTensor:\n"
        "    def __init__(self, requires_grad=False):\n"
        "        self.requires_grad = requires_grad\n"
        "\n"
        "# === Single op, grad input, all gates True → True ===\n"
        "x = MiniTensor(requires_grad=True)\n"
        "table = [{'op_inputs': [0], 'is_differentiable': True, 'grad_tracking_enabled': True}]\n"
        "assert ex3_chain_propagate([x], table) is True\n"
        "\n"
        "# === Single op, no grad input → False ===\n"
        "x = MiniTensor(requires_grad=False)\n"
        "table = [{'op_inputs': [0], 'is_differentiable': True, 'grad_tracking_enabled': True}]\n"
        "assert ex3_chain_propagate([x], table) is False\n"
        "\n"
        "# === Single op, grad input but toggle off → False ===\n"
        "x = MiniTensor(requires_grad=True)\n"
        "table = [{'op_inputs': [0], 'is_differentiable': True, 'grad_tracking_enabled': False}]\n"
        "assert ex3_chain_propagate([x], table) is False\n"
        "\n"
        "# === Single op, grad input but not differentiable → False ===\n"
        "x = MiniTensor(requires_grad=True)\n"
        "table = [{'op_inputs': [0], 'is_differentiable': False, 'grad_tracking_enabled': True}]\n"
        "assert ex3_chain_propagate([x], table) is False\n"
        "\n"
        "# === Two-op chain: rg propagates from step 1 to step 2 ===\n"
        "x = MiniTensor(requires_grad=True)\n"
        "table = [\n"
        "    {'op_inputs': [0], 'is_differentiable': True, 'grad_tracking_enabled': True},\n"
        "    {'op_inputs': [0], 'is_differentiable': True, 'grad_tracking_enabled': True},\n"
        "]\n"
        "assert ex3_chain_propagate([x], table) is True\n"
        "\n"
        "# === Chain SNAPS at non-differentiable op ===\n"
        "x = MiniTensor(requires_grad=True)\n"
        "table = [\n"
        "    {'op_inputs': [0], 'is_differentiable': True, 'grad_tracking_enabled': True},\n"
        "    {'op_inputs': [0], 'is_differentiable': False, 'grad_tracking_enabled': True},  # SNAP\n"
        "    {'op_inputs': [0], 'is_differentiable': True, 'grad_tracking_enabled': True},\n"
        "]\n"
        "assert ex3_chain_propagate([x], table) is False, 'non-diff op must snap the chain'\n"
        "\n"
        "# === Mixed inputs: only one needs requires_grad ===\n"
        "x = MiniTensor(requires_grad=True)\n"
        "y = MiniTensor(requires_grad=False)\n"
        "table = [{'op_inputs': [0, 1], 'is_differentiable': True, 'grad_tracking_enabled': True}]\n"
        "assert ex3_chain_propagate([x, y], table) is True\n"
        "\n"
        "# === All inputs grad-free → False even if differentiable ===\n"
        "x = MiniTensor(requires_grad=False)\n"
        "y = MiniTensor(requires_grad=False)\n"
        "table = [{'op_inputs': [0, 1], 'is_differentiable': True, 'grad_tracking_enabled': True}]\n"
        "assert ex3_chain_propagate([x, y], table) is False\n"
        "\n"
        "# === Empty op_table → False (no output to flag) ===\n"
        "assert ex3_chain_propagate([MiniTensor(requires_grad=True)], []) is False\n"
        "\n"
        "# === Long chain: 5 differentiable ops with grad input → still True ===\n"
        "x = MiniTensor(requires_grad=True)\n"
        "table = [{'op_inputs': [0], 'is_differentiable': True, 'grad_tracking_enabled': True}] * 5\n"
        "assert ex3_chain_propagate([x], table) is True\n"
        "\n"
        "# === Toggle off at any step in the chain → False ===\n"
        "x = MiniTensor(requires_grad=True)\n"
        "table = [\n"
        "    {'op_inputs': [0], 'is_differentiable': True, 'grad_tracking_enabled': True},\n"
        "    {'op_inputs': [0], 'is_differentiable': True, 'grad_tracking_enabled': False},  # turn off\n"
        "    {'op_inputs': [0], 'is_differentiable': True, 'grad_tracking_enabled': True},\n"
        "]\n"
        "assert ex3_chain_propagate([x], table) is False, 'toggle-off step must propagate False forward'"
    ),
    "solution_body": (
        "def ex3_chain_propagate(inputs, op_table):\n"
        "    if not op_table:\n"
        "        return False\n"
        "    # The first step indexes into the original `inputs` list.\n"
        "    current = list(inputs)\n"
        "    out = None\n"
        "    for step in op_table:\n"
        "        step_inputs = [current[i] for i in step['op_inputs']]\n"
        "        any_input_rg = any(getattr(a, 'requires_grad', False) for a in step_inputs)\n"
        "        out_rg = (\n"
        "            step['grad_tracking_enabled']\n"
        "            and step['is_differentiable']\n"
        "            and any_input_rg\n"
        "        )\n"
        "        # Build a tiny dummy with the propagated flag.\n"
        "        out_obj = type('Out', (), {'requires_grad': out_rg})()\n"
        "        current = [out_obj]  # next step indexes into a 1-tensor list\n"
        "        out = out_obj\n"
        "    return out.requires_grad"
    ),
    "solution_notes": (
        "**Three-gate AND at each step is the entire invariant.** No "
        "extra state — just apply ex1's rule to the step's inputs, store "
        "the result, feed it forward. Composition emerges from "
        "iteration; you don't need a graph data structure for this drill.\n\n"
        "**Once the chain SNAPS, it stays snapped.** The propagation is "
        "ASYMMETRIC: once a step outputs `requires_grad=False`, every "
        "downstream step sees that False as its only input and outputs "
        "False too (since `any(False) == False`). PyTorch's own "
        "autograd matches this — once you `.detach()`, no downstream op "
        "can re-attach the gradient.\n\n"
        "**Toggle changes are per-op.** The drill's `grad_tracking_"
        "enabled` field is per-step because the toggle can flip "
        "between ops (entering/exiting a `no_grad()` block). A step's "
        "output requires_grad depends on the toggle AT THAT STEP, not "
        "at the start of the chain."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 8 — unbroadcast-pattern ex3 (add_back0/1 uses unbroadcast)
# ---------------------------------------------------------------------------

SPEC_UNBROADCAST = {
    "atom_id": "unbroadcast-pattern",
    "subtopic": "Backprop: Unbroadcast pattern",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_UNBROADCAST_IN_BACK,
    "exercise_index": 3,
    "exercise_title": "add_back0 / add_back1 — wire unbroadcast into a broadcasting binary back fn",
    "slug": "add-back-wires-unbroadcast-into-broadcasting-binary-back-fn",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["unbroadcast", "add-back", "binary-op", "broadcasting"],
    "kcs": [
        "unbroadcast-pattern",
        "binary-back-fn-restores-input-shape",
    ],
    "lo": (
        "Apply the unbroadcast pattern inside `add_back0` and `add_back1` "
        "so that even when `x + y` broadcast `x` against `y`, each "
        "returned grad has shape matching its respective input — bridging "
        "the elementwise math gradient and the input-shape contract."
    ),
    "prompt_body": (
        "Implement `ex3_add_back()` returning a dict with three keys: "
        "`'unbroadcast'`, `'add_back0'`, `'add_back1'`.\n\n"
        "1. `unbroadcast(grad, original)` — given a `grad` whose shape "
        "is the broadcasted output shape, sum it back down to "
        "`original.shape`:\n"
        "   - Step A: while `grad.ndim > original.ndim`, `grad = "
        "grad.sum(dim=0)`.\n"
        "   - Step B: for each axis `i` where `original.shape[i] == 1` "
        "and `grad.shape[i] != 1`, `grad = grad.sum(dim=i, keepdim=True)`.\n"
        "   - Return the result. Final shape must equal `original.shape`.\n\n"
        "2. `add_back0(grad_out, out, x, y) -> grad_x` — for `out = x + y` "
        "(possibly broadcasting). Math gradient is `1`, so locally "
        "`grad_x = grad_out`, then unbroadcast to `x.shape`.\n\n"
        "3. `add_back1(grad_out, out, x, y) -> grad_y` — symmetric: "
        "`grad_y = grad_out`, then unbroadcast to `y.shape`.\n\n"
        "Constraint: `add_back0` and `add_back1` MUST call your "
        "`unbroadcast` helper — do not inline its body in either of them."
    ),
    "stub": (
        "def ex3_add_back():\n"
        '    """Return {unbroadcast, add_back0, add_back1} — binary back fns using unbroadcast."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "result = ex3_add_back()\n"
        "unbroadcast = result['unbroadcast']\n"
        "add_back0 = result['add_back0']\n"
        "add_back1 = result['add_back1']\n"
        "\n"
        "# === unbroadcast: leading-axes only ===\n"
        "g = t.ones(2, 3, 4)\n"
        "orig = t.zeros(3, 4)\n"
        "out = unbroadcast(g, orig)\n"
        "assert out.shape == orig.shape, f'leading-only: shape {out.shape}'\n"
        "assert t.allclose(out, t.full((3, 4), 2.0))\n"
        "\n"
        "# === unbroadcast: size-1 only ===\n"
        "g = t.ones(3, 5, 4)\n"
        "orig = t.zeros(3, 1, 4)\n"
        "out = unbroadcast(g, orig)\n"
        "assert out.shape == orig.shape\n"
        "assert t.allclose(out, t.full((3, 1, 4), 5.0))\n"
        "\n"
        "# === unbroadcast: combined leading + size-1 ===\n"
        "g = t.ones(2, 3, 1, 4)\n"
        "orig = t.zeros(3, 1, 1)\n"
        "out = unbroadcast(g, orig)\n"
        "assert out.shape == orig.shape, f'combined: got {out.shape}'\n"
        "# 2 leading axes (2*1=2) * 4 (broadcast in axis 2 of orig if orig.shape[1]==1)\n"
        "# Easier: just verify shape contract.\n"
        "\n"
        "# === add_back: no broadcasting (same shape both inputs) ===\n"
        "x = t.tensor([[1.0, 2.0], [3.0, 4.0]])\n"
        "y = t.tensor([[10.0, 20.0], [30.0, 40.0]])\n"
        "out = x + y\n"
        "grad_out = t.ones_like(out)\n"
        "g0 = add_back0(grad_out, out, x, y)\n"
        "g1 = add_back1(grad_out, out, x, y)\n"
        "assert g0.shape == x.shape\n"
        "assert g1.shape == y.shape\n"
        "assert t.allclose(g0, t.ones_like(x))\n"
        "assert t.allclose(g1, t.ones_like(y))\n"
        "\n"
        "# === add_back: leading-axis broadcast ===\n"
        "# x.shape == (4,), y.shape == (3, 4) → out.shape == (3, 4).\n"
        "# grad_x must be summed across leading axis 0 to get back to (4,).\n"
        "x = t.ones(4)\n"
        "y = t.ones(3, 4)\n"
        "out = x + y\n"
        "assert out.shape == (3, 4)\n"
        "grad_out = t.ones_like(out)\n"
        "g0 = add_back0(grad_out, out, x, y)\n"
        "g1 = add_back1(grad_out, out, x, y)\n"
        "assert g0.shape == x.shape, f'g0 shape: {g0.shape}, expected {x.shape}'\n"
        "assert g1.shape == y.shape, f'g1 shape: {g1.shape}, expected {y.shape}'\n"
        "assert t.allclose(g0, t.full((4,), 3.0))  # summed 3 leading rows\n"
        "assert t.allclose(g1, t.ones(3, 4))\n"
        "\n"
        "# === add_back: size-1 broadcast ===\n"
        "# x.shape == (3, 1), y.shape == (3, 5) → out.shape == (3, 5).\n"
        "x = t.ones(3, 1)\n"
        "y = t.ones(3, 5)\n"
        "out = x + y\n"
        "grad_out = t.ones_like(out)\n"
        "g0 = add_back0(grad_out, out, x, y)\n"
        "g1 = add_back1(grad_out, out, x, y)\n"
        "assert g0.shape == (3, 1), f'g0 shape: {g0.shape}'\n"
        "assert g1.shape == (3, 5)\n"
        "assert t.allclose(g0, t.full((3, 1), 5.0))  # summed 5 cols\n"
        "\n"
        "# === Cross-check vs torch.autograd ===\n"
        "x = t.randn(4, requires_grad=True)\n"
        "y = t.randn(3, 4, requires_grad=True)\n"
        "out_t = x + y\n"
        "out_t.sum().backward()\n"
        "ours_x = add_back0(t.ones_like(out_t), out_t.detach(), x.detach(), y.detach())\n"
        "ours_y = add_back1(t.ones_like(out_t), out_t.detach(), x.detach(), y.detach())\n"
        "assert t.allclose(x.grad, ours_x), f'autograd x.grad mismatch: {x.grad} vs {ours_x}'\n"
        "assert t.allclose(y.grad, ours_y)\n"
        "\n"
        "# === Combined broadcast (leading + size-1) — the headline case ===\n"
        "x = t.ones(1, 4)\n"
        "y = t.ones(2, 3, 4)\n"
        "out = x + y\n"
        "assert out.shape == (2, 3, 4)\n"
        "grad_out = t.ones_like(out)\n"
        "g0 = add_back0(grad_out, out, x, y)\n"
        "g1 = add_back1(grad_out, out, x, y)\n"
        "assert g0.shape == x.shape, f'g0 shape: {g0.shape}, expected {x.shape}'\n"
        "assert g1.shape == y.shape\n"
        "# After 2 leading-axis peels (2*3=6 rows), shape (1,4). All 6 contributed.\n"
        "assert t.allclose(g0, t.full((1, 4), 6.0))\n"
        "assert t.allclose(g1, t.ones(2, 3, 4))"
    ),
    "solution_body": (
        "def ex3_add_back():\n"
        "    def unbroadcast(grad, original):\n"
        "        # Step A: peel leading axes.\n"
        "        while grad.ndim > original.ndim:\n"
        "            grad = grad.sum(dim=0)\n"
        "        # Step B: collapse size-1 axes that got expanded.\n"
        "        for i, (g_dim, o_dim) in enumerate(zip(grad.shape, original.shape)):\n"
        "            if o_dim == 1 and g_dim != 1:\n"
        "                grad = grad.sum(dim=i, keepdim=True)\n"
        "        return grad\n"
        "\n"
        "    def add_back0(grad_out, out, x, y):\n"
        "        # d(x+y)/dx = 1 → local grad is grad_out, unbroadcast to x.shape.\n"
        "        return unbroadcast(grad_out, x)\n"
        "\n"
        "    def add_back1(grad_out, out, x, y):\n"
        "        # d(x+y)/dy = 1 → local grad is grad_out, unbroadcast to y.shape.\n"
        "        return unbroadcast(grad_out, y)\n"
        "\n"
        "    return {\n"
        "        'unbroadcast': unbroadcast,\n"
        "        'add_back0': add_back0,\n"
        "        'add_back1': add_back1,\n"
        "    }"
    ),
    "solution_notes": (
        "**Order of unbroadcast steps matters.** Peel leading axes FIRST "
        "(step A), then collapse size-1 axes (step B). The reverse order "
        "would mis-align dims: a size-1 axis index in `original.shape` "
        "doesn't necessarily match the same index in the larger "
        "`grad.shape` until you've peeled the leading dims off.\n\n"
        "**`add_back0` and `add_back1` are the same shape.** Both "
        "have local gradient `1`, so both return "
        "`unbroadcast(grad_out, x_or_y)`. This is the ONLY binary op "
        "where the two back fns are functionally identical — "
        "`mul_back0` is `unbroadcast(grad_out * y, x)`, `div_back0` is "
        "`unbroadcast(grad_out / y, x)`, etc. Add is the easiest case to "
        "introduce the pattern.\n\n"
        "**Composability with the parents-dispatch drill.** Once "
        "`add_back0` / `add_back1` are registered into `BACK_FUNCS` at "
        "`(t.add, 0)` and `(t.add, 1)`, the dispatcher from the "
        "parents-dispatch drill calls them automatically. The whole "
        "pattern composes — no special-casing for broadcasting at the "
        "dispatcher level."
    ),
    "extra_imports": [],
}


SPECS = [
    SPEC_ARGPOS,
    SPEC_CHAIN,
    SPEC_TOGGLE,
    SPEC_KWARGS,
    SPEC_PARENTS,
    SPEC_RECIPE,
    SPEC_REQGRAD,
    SPEC_UNBROADCAST,
]


# ---------------------------------------------------------------------------
# Verifier — execute stub, solution, then tests to catch errors locally.
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
    print(f"[deepening_c_batch14] Verifying {len(SPECS)} specs...")
    _verify_all(SPECS)

    print(f"\n[deepening_c_batch14] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_c_batch14] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
