#!/usr/bin/env python3
"""Author Colab-native standalones for the ARENA part 4 backprop-wrapper atoms.

Eight standalones, brand-new `prereqs_backprop/` folder (no prior splits):

  * backward-fn-signature        — ex1, ex2
  * register-back-fn-after-wrap  — ex1, ex2
  * wrap-forward-fn-generic      — ex1, ex2
  * param-grad-access            — ex1
  * buffer-copy_-inplace         — ex1

Each drill exercises the MANUAL backprop pattern from ARENA part 4 — a tiny
hand-written `Tensor` wrapper class plus a `BACK_FUNCS` lookup dict plus
`wrap_forward_fn`-style closures — without calling `torch.autograd`. The drill
tests use plain `torch.Tensor` for shape/value math only.

Smaller-than-ARENA scope: ARENA's `Tensor.__mul__` + `multiply_back0/back1` is
the big composite. Each drill below picks ONE constituent skill (a single
backward fn, a single BACK_FUNCS entry, the generic wrap shell, etc).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_backprop"


# ---------------------------------------------------------------- atom recaps

RECAP_BACKWARD_FN_SIG = (
    "## Backward-fn signature — quick refresher\n"
    "\n"
    "In a manual autograd, every forward op `f(x, y, ...) -> out` is paired "
    "with **one backward fn per input position**. The canonical signature is:\n"
    "\n"
    "```python\n"
    "def f_back<i>(grad_out, out, *args, **kwargs):\n"
    "    \"\"\"dL/dargs[i] given dL/dout, cached out, and the original args.\"\"\"\n"
    "    ...\n"
    "```\n"
    "\n"
    "- `grad_out` — the upstream gradient `dL/dout`, same shape as `out`.\n"
    "- `out` — the cached forward output (so you don't recompute).\n"
    "- `*args, **kwargs` — the original forward inputs (one of them is the "
    "input you're differentiating w.r.t.).\n"
    "\n"
    "The fn returns `dL/dargs[i]`, **same shape as `args[i]`**. For elementwise "
    "ops the local Jacobian is diagonal so you just multiply `grad_out` by the "
    "elementwise derivative — no actual matrix is materialized."
)

RECAP_REGISTER_BACK_FN = (
    "## Register-back-fn-after-wrap — quick refresher\n"
    "\n"
    "A tiny autograd needs to look up *which* backward fn corresponds to *which* "
    "forward op at *which* argument position. The convention is a dict keyed by "
    "`(forward_fn, argnum)`:\n"
    "\n"
    "```python\n"
    "class BackwardFuncLookup:\n"
    "    def __init__(self): self._table = {}\n"
    "    def add_back_func(self, fwd, argnum, back_fn):\n"
    "        self._table[(fwd, argnum)] = back_fn\n"
    "    def get_back_func(self, fwd, argnum):\n"
    "        return self._table[(fwd, argnum)]\n"
    "\n"
    "BACK_FUNCS = BackwardFuncLookup()\n"
    "log = wrap_forward_fn(torch.log)\n"
    "BACK_FUNCS.add_back_func(torch.log, 0, log_back)   # the wrap+register pair\n"
    "```\n"
    "\n"
    "Two steps, always in this order: (1) wrap the forward fn so it builds a "
    "Recipe; (2) register the backward fn so the reverse pass can find it. "
    "Binary ops register TWICE — once for argnum=0, once for argnum=1."
)

RECAP_WRAP_FORWARD_FN = (
    "## wrap_forward_fn — quick refresher\n"
    "\n"
    "`wrap_forward_fn` is the **factory** that turns a plain numerical fn "
    "(`torch.log`, `torch.multiply`, ...) into an autograd-aware version that "
    "(a) unboxes Tensor → raw, (b) runs the forward, (c) boxes the result back "
    "into a Tensor, and (d) attaches a Recipe so the reverse pass can replay "
    "the call.\n"
    "\n"
    "```python\n"
    "def wrap_forward_fn(fwd_fn):\n"
    "    def tensor_func(*args, **kwargs):\n"
    "        raw_args = tuple(a.array if isinstance(a, Tensor) else a for a in args)\n"
    "        out_raw = fwd_fn(*raw_args, **kwargs)\n"
    "        out = Tensor(out_raw)\n"
    "        # if any input is a tracked Tensor, attach a Recipe\n"
    "        out.recipe = Recipe(fwd_fn, raw_args, kwargs, parents)\n"
    "        return out\n"
    "    return tensor_func\n"
    "```\n"
    "\n"
    "The closure captures `fwd_fn` — one factory call replaces dozens of "
    "hand-written wrappers. Every wrapper shares the same unbox-call-box "
    "skeleton; only `fwd_fn` varies."
)

RECAP_PARAM_GRAD = (
    "## param.grad access — quick refresher\n"
    "\n"
    "In PyTorch (and any clone of its API), a `Parameter` is a `Tensor` "
    "subclass that *participates in backprop*. After `loss.backward()`, each "
    "parameter holds its accumulated gradient on the `.grad` attribute, "
    "matching the parameter's shape:\n"
    "\n"
    "```python\n"
    "for p in model.parameters():\n"
    "    print(p.shape, p.grad.shape)   # always equal\n"
    "    p.data -= lr * p.grad           # SGD step\n"
    "    p.grad = None                   # zero_grad: drop the tensor\n"
    "```\n"
    "\n"
    "- `p.grad is None` BEFORE any backward pass — guard against this.\n"
    "- `p.grad` is **accumulated**: every backward pass *adds* to it (so you "
    "must zero it between steps).\n"
    "- The standard zero strategy is `p.grad = None`, not `p.grad.zero_()` — "
    "saves memory and matches PyTorch's default in `optimizer.zero_grad("
    "set_to_none=True)`."
)

RECAP_BUFFER_COPY = (
    "## in-place buffer copy_ — quick refresher\n"
    "\n"
    "Modules like `BatchNorm` track running statistics (mean, variance) as "
    "**buffers** — tensors that are part of the module state but DO NOT receive "
    "gradients. The canonical update pattern is `running_buf.copy_(new_value)`:\n"
    "\n"
    "```python\n"
    "self.running_mean.copy_(\n"
    "    (1 - momentum) * self.running_mean + momentum * batch_mean\n"
    ")\n"
    "```\n"
    "\n"
    "Why `copy_` and not `=`?\n"
    "- `self.running_mean = new_tensor` would REBIND the attribute, losing the "
    "registered-buffer link (so `.to(device)`, `.state_dict()`, etc. would "
    "stop tracking it).\n"
    "- `self.running_mean.copy_(new_tensor)` writes into the existing storage. "
    "Identity (`id`) is preserved.\n"
    "\n"
    "`copy_` accepts any broadcastable tensor and ignores requires_grad on the "
    "source — exactly what we want for non-differentiable state updates."
)


# ---------------------------------------------------------------- spec helpers

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
) -> dict:
    dots = ("🔴" * difficulty_num) + ("⚪" * (5 - difficulty_num))
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
        "extra_imports": [],
    }


# =========================================================================
# atom: backward-fn-signature  (2 exercises)
# =========================================================================

SPEC_BFS_1 = _spec(
    atom_id="backward-fn-signature",
    subtopic="Backprop: backward fn signature",
    recap=RECAP_BACKWARD_FN_SIG,
    ex_idx=1,
    ex_title="write log_back with the canonical (grad_out, out, x) signature",
    slug="write-log-back-with-canonical-signature",
    bloom="Apply",
    difficulty_num=2,
    keywords=["backward-fn", "log", "elementwise-derivative", "signature"],
    kcs=["backward-fn-signature", "chain-rule-elementwise"],
    lo=(
        "Apply the (grad_out, out, x) backward-fn convention by writing "
        "log_back, the per-element gradient of torch.log."
    ),
    prompt_body=(
        "Implement `log_back(grad_out, out, x)`. This is the simplest "
        "elementwise backward fn — the per-position warm-up for the whole "
        "BACK_FUNCS table.\n\n"
        "**The math.** `out = log(x)` ⇒ `d(out)/d(x) = 1/x` elementwise. By "
        "the chain rule, `dL/dx = dL/dout * d(out)/d(x) = grad_out / x`.\n\n"
        "**The signature.** All ARENA back fns share the shape "
        "`(grad_out, out, *args, **kwargs) -> grad_in`:\n\n"
        "- `grad_out` — upstream gradient `dL/dout`, same shape as `out`.\n"
        "- `out` — the cached forward result `log(x)` (you don't need it for "
        "  log_back, but it's part of the contract).\n"
        "- `x` — the original input to `log`.\n\n"
        "Return `dL/dx` with the **same shape and dtype as `x`**.\n\n"
        "Inputs are `torch.Tensor` for this drill; no autograd, no grad "
        "tracking — just elementwise tensor arithmetic."
    ),
    stub=(
        "def log_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        '    """Gradient of out = log(x). Returns dL/dx given dL/dout."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# Scalar sanity check: d(log x)/dx = 1/x, so grad_out=1, x=2 → 0.5\n"
        "x = t.tensor([2.0])\n"
        "out = t.log(x)\n"
        "g = log_back(t.tensor([1.0]), out, x)\n"
        "assert t.allclose(g, t.tensor([0.5])), f'scalar fail: got {g}'\n"
        "\n"
        "# Vector — shape preserved, value = grad_out / x.\n"
        "x = t.tensor([1.0, 2.0, 4.0, 8.0])\n"
        "out = t.log(x)\n"
        "grad_out = t.tensor([1.0, 1.0, 1.0, 1.0])\n"
        "g = log_back(grad_out, out, x)\n"
        "expected = t.tensor([1.0, 0.5, 0.25, 0.125])\n"
        "assert g.shape == x.shape, f'shape mismatch: {g.shape} vs {x.shape}'\n"
        "assert t.allclose(g, expected), f'vector fail: got {g}'\n"
        "\n"
        "# Non-trivial grad_out — chain rule scales each entry independently.\n"
        "x = t.tensor([1.0, 2.0, 5.0])\n"
        "out = t.log(x)\n"
        "grad_out = t.tensor([3.0, -2.0, 10.0])\n"
        "g = log_back(grad_out, out, x)\n"
        "expected = t.tensor([3.0 / 1.0, -2.0 / 2.0, 10.0 / 5.0])\n"
        "assert t.allclose(g, expected), f'chain rule fail: got {g}'\n"
        "\n"
        "# Matrix shape — must broadcast / pass through unchanged.\n"
        "x = t.tensor([[1.0, 2.0], [4.0, 8.0]])\n"
        "out = t.log(x)\n"
        "grad_out = t.ones_like(x)\n"
        "g = log_back(grad_out, out, x)\n"
        "assert g.shape == (2, 2), f'matrix shape mismatch: {g.shape}'\n"
        "assert t.allclose(g, 1.0 / x), 'matrix value fail'\n"
        "\n"
        "# Cross-check against torch.autograd (purely as a witness — we don't "
        "use it in our backward fn).\n"
        "x_ref = t.tensor([1.5, 3.5, 7.5], requires_grad=True)\n"
        "y_ref = t.log(x_ref).sum()\n"
        "y_ref.backward()\n"
        "g_ours = log_back(t.ones(3), t.log(x_ref.detach()), x_ref.detach())\n"
        "assert t.allclose(g_ours, x_ref.grad), (\n"
        "    f'mismatch vs autograd: ours={g_ours}, ref={x_ref.grad}'\n"
        ")"
    ),
    solution_body=(
        "def log_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        "    # d(log x)/dx = 1/x; chain rule → grad_out / x.\n"
        "    # We never read `out` — but it's part of the contract so every\n"
        "    # back fn has the same call shape (uniform dispatch).\n"
        "    return grad_out / x"
    ),
    solution_notes=(
        "**Why `out` even though we don't use it.** Uniform calling convention. "
        "The reverse pass dispatches `back_fn(grad_out, node.array, *node.recipe.args)` "
        "for every node, regardless of which fn it is. `exp_back` will need `out` "
        "(since `d(exp x)/dx = exp(x) = out`); `log_back` doesn't. Keeping the "
        "signature uniform means one dispatcher, no special cases.\n\n"
        "**Elementwise = diagonal Jacobian.** For elementwise ops, the Jacobian is "
        "diagonal so the chain rule reduces to elementwise multiplication. No matrix "
        "is materialized. This is why log/exp/relu/etc. backward fns are one-liners."
    ),
)

SPEC_BFS_2 = _spec(
    atom_id="backward-fn-signature",
    subtopic="Backprop: backward fn signature",
    recap=RECAP_BACKWARD_FN_SIG,
    ex_idx=2,
    ex_title="write negative_back and exp_back — back-fn signature, two ops",
    slug="negative-and-exp-back-two-ops",
    bloom="Apply",
    difficulty_num=2,
    keywords=["backward-fn", "negative", "exp", "uniform-signature"],
    kcs=["backward-fn-signature", "back-fn-uses-cached-out"],
    lo=(
        "Apply the uniform (grad_out, out, x) backward-fn signature across two "
        "ops — one that ignores `out`, one that requires it — to internalize the "
        "calling convention."
    ),
    prompt_body=(
        "Implement TWO backward fns that share the same signature:\n\n"
        "**1. `negative_back(grad_out, out, x)`** — gradient of `out = -x`.\n"
        "   - Math: `d(-x)/dx = -1`, so `dL/dx = -grad_out`.\n"
        "   - Notice you don't need `x` OR `out` — pure sign flip of grad_out.\n\n"
        "**2. `exp_back(grad_out, out, x)`** — gradient of `out = exp(x)`.\n"
        "   - Math: `d(exp x)/dx = exp(x) = out`, so `dL/dx = grad_out * out`.\n"
        "   - This is the contrasting case — `out` IS used (and faster than "
        "     re-computing `exp(x)`).\n\n"
        "**The point of this drill.** Both fns take `(grad_out, out, x)`. One "
        "ignores most of its inputs, the other uses `out` directly. The "
        "uniform calling convention is what makes the BACK_FUNCS dispatcher "
        "work — no signature gymnastics.\n\n"
        "Return tensors with the SAME shape as `x` for both fns."
    ),
    stub=(
        "def negative_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        '    """Gradient of out = -x. Returns dL/dx."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def exp_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        '    """Gradient of out = exp(x). Returns dL/dx."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- negative_back ---\n"
        "x = t.tensor([1.0, -2.0, 3.5])\n"
        "out = -x\n"
        "grad_out = t.tensor([5.0, 7.0, 11.0])\n"
        "g = negative_back(grad_out, out, x)\n"
        "expected = t.tensor([-5.0, -7.0, -11.0])\n"
        "assert g.shape == x.shape, f'negative_back shape: {g.shape}'\n"
        "assert t.allclose(g, expected), f'negative_back value: {g}'\n"
        "\n"
        "# negative_back should NOT depend on x or out (sanity).\n"
        "x_alt = t.tensor([99.0, -99.0, 0.0])\n"
        "g_alt = negative_back(grad_out, -x_alt, x_alt)\n"
        "assert t.allclose(g_alt, expected), (\n"
        "    'negative_back must depend only on grad_out — got value drift'\n"
        ")\n"
        "\n"
        "# --- exp_back ---\n"
        "x = t.tensor([0.0, 1.0, 2.0])\n"
        "out = t.exp(x)  # = [1, e, e^2]\n"
        "grad_out = t.ones(3)\n"
        "g = exp_back(grad_out, out, x)\n"
        "assert g.shape == x.shape, f'exp_back shape: {g.shape}'\n"
        "assert t.allclose(g, out), f'exp_back must equal out when grad_out=1: {g}'\n"
        "\n"
        "# exp_back with non-unit grad_out — must scale by out.\n"
        "x = t.tensor([1.0, 2.0])\n"
        "out = t.exp(x)\n"
        "grad_out = t.tensor([3.0, -4.0])\n"
        "g = exp_back(grad_out, out, x)\n"
        "expected = grad_out * out\n"
        "assert t.allclose(g, expected), f'exp_back chain rule: {g}'\n"
        "\n"
        "# Matrix shape for both.\n"
        "X = t.tensor([[1.0, 2.0], [-1.0, -2.0]])\n"
        "G = t.ones_like(X)\n"
        "neg_out = -X\n"
        "exp_out = t.exp(X)\n"
        "g_neg = negative_back(G, neg_out, X)\n"
        "g_exp = exp_back(G, exp_out, X)\n"
        "assert g_neg.shape == (2, 2) and g_exp.shape == (2, 2)\n"
        "assert t.allclose(g_neg, -G)\n"
        "assert t.allclose(g_exp, exp_out)\n"
        "\n"
        "# Witness vs torch.autograd.\n"
        "for fn_name, fwd, back_fn in [\n"
        "    ('negative', lambda u: -u, negative_back),\n"
        "    ('exp', lambda u: t.exp(u), exp_back),\n"
        "]:\n"
        "    x_ref = t.tensor([0.5, 1.5, -0.5], requires_grad=True)\n"
        "    y_ref = fwd(x_ref).sum()\n"
        "    y_ref.backward()\n"
        "    out_cached = fwd(x_ref.detach())\n"
        "    g_ours = back_fn(t.ones(3), out_cached, x_ref.detach())\n"
        "    assert t.allclose(g_ours, x_ref.grad), (\n"
        "        f'{fn_name}_back disagrees with autograd: '\n"
        "        f'ours={g_ours}, ref={x_ref.grad}'\n"
        "    )"
    ),
    solution_body=(
        "def negative_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        "    # d(-x)/dx = -1, so dL/dx = -grad_out. `out` and `x` unused.\n"
        "    return -grad_out\n"
        "\n"
        "\n"
        "def exp_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        "    # d(exp x)/dx = exp(x) = out (cached). dL/dx = grad_out * out.\n"
        "    return grad_out * out"
    ),
    solution_notes=(
        "**The two halves of why `out` is in the signature.** `negative_back` "
        "ignores `out`. `exp_back` *requires* `out` (and benefits from it being "
        "cached — no second `exp` call). Most back fns fall into one of these "
        "two buckets. Some, like `sigmoid_back`, even use `out` instead of `x` "
        "because the formula is cleaner: `d/dx sigmoid(x) = sigmoid(x)*(1 - "
        "sigmoid(x)) = out * (1 - out)`.\n\n"
        "**Why a uniform signature beats per-op signatures.** ARENA's reverse "
        "pass is one line: `back_fn(grad_out, node.array, *node.recipe.args, "
        "**node.recipe.kwargs)`. If each back fn had a bespoke signature, the "
        "dispatcher would need switch logic per op — at which point you've "
        "rebuilt torch.autograd's C++ dispatcher in slow Python."
    ),
)


# =========================================================================
# atom: register-back-fn-after-wrap  (2 exercises)
# =========================================================================

SPEC_RBF_1 = _spec(
    atom_id="register-back-fn-after-wrap",
    subtopic="Backprop: register back fn",
    recap=RECAP_REGISTER_BACK_FN,
    ex_idx=1,
    ex_title="wire one entry into BACK_FUNCS and dispatch it",
    slug="wire-one-entry-into-back-funcs",
    bloom="Apply",
    difficulty_num=2,
    keywords=["back-funcs", "dict-by-tuple", "register", "dispatch"],
    kcs=["register-back-fn-after-wrap", "backward-func-lookup"],
    lo=(
        "Apply the (forward_fn, argnum) → back_fn registration pattern by "
        "wiring a single entry into a BackwardFuncLookup table and dispatching "
        "through it."
    ),
    prompt_body=(
        "Implement TWO things:\n\n"
        "**1. `BackwardFuncLookup` class** with two methods:\n"
        "   - `add_back_func(fwd_fn, argnum, back_fn)` — register a back fn.\n"
        "   - `get_back_func(fwd_fn, argnum)` — look one up.\n"
        "   Use a dict keyed by `(fwd_fn, argnum)` tuples internally.\n\n"
        "**2. `register_log(BACK_FUNCS)`** — given a fresh lookup table, "
        "register the backward fn for `torch.log` at argnum=0. The backward fn "
        "is provided to you as `log_back(grad_out, out, x) -> grad_out / x`.\n\n"
        "The test cell then **dispatches** through the table: it looks up "
        "`(torch.log, 0)` and calls the returned back fn with concrete args, "
        "verifying both the registration and the lookup work.\n\n"
        "**Why this is the bedrock of the autograd dispatcher.** Once the "
        "table is populated, the reverse pass becomes a generic loop: pop a "
        "Tensor off the topo-sorted queue, look up its `(recipe.func, argnum)`, "
        "call the back fn. No `if fn == log: ...` chains. This drill is the "
        "single-entry baby step."
    ),
    stub=(
        "class BackwardFuncLookup:\n"
        '    """Dict-keyed-by-(fwd_fn, argnum) lookup table for backward fns."""\n'
        "    def __init__(self):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    def add_back_func(self, fwd_fn, argnum, back_fn):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    def get_back_func(self, fwd_fn, argnum):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "\n"
        "def log_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        "    return grad_out / x\n"
        "\n"
        "\n"
        "def register_log(BACK_FUNCS: 'BackwardFuncLookup') -> None:\n"
        '    """Register log_back for torch.log at argnum=0."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- lookup table sanity ---\n"
        "tbl = BackwardFuncLookup()\n"
        "\n"
        "def fake_back(grad_out, out, x):\n"
        "    return grad_out * 2\n"
        "\n"
        "tbl.add_back_func(t.sin, 0, fake_back)\n"
        "got = tbl.get_back_func(t.sin, 0)\n"
        "assert got is fake_back, f'add/get round-trip failed: got {got}'\n"
        "\n"
        "# Different argnums for the same fwd_fn must be independent slots.\n"
        "def fake_back_arg1(grad_out, out, x, y):\n"
        "    return grad_out * 3\n"
        "\n"
        "tbl.add_back_func(t.multiply, 0, fake_back)\n"
        "tbl.add_back_func(t.multiply, 1, fake_back_arg1)\n"
        "assert tbl.get_back_func(t.multiply, 0) is fake_back\n"
        "assert tbl.get_back_func(t.multiply, 1) is fake_back_arg1\n"
        "\n"
        "# --- register_log + dispatch ---\n"
        "BACK_FUNCS = BackwardFuncLookup()\n"
        "register_log(BACK_FUNCS)\n"
        "\n"
        "# Pull the registered fn back out — must equal log_back.\n"
        "fn = BACK_FUNCS.get_back_func(t.log, 0)\n"
        "assert fn is log_back, f'expected log_back, got {fn}'\n"
        "\n"
        "# End-to-end dispatch: simulate the reverse pass for a single log node.\n"
        "x = t.tensor([1.0, 2.0, 4.0])\n"
        "out = t.log(x)\n"
        "grad_out = t.ones(3)\n"
        "# This is what backprop's inner loop does:\n"
        "back_fn = BACK_FUNCS.get_back_func(t.log, 0)\n"
        "grad_x = back_fn(grad_out, out, x)\n"
        "expected = t.tensor([1.0, 0.5, 0.25])\n"
        "assert t.allclose(grad_x, expected), f'dispatch fail: {grad_x}'\n"
        "\n"
        "# Looking up an unregistered (fn, argnum) must raise — silent return\n"
        "# would mask bugs in real backprop.\n"
        "try:\n"
        "    BACK_FUNCS.get_back_func(t.exp, 0)\n"
        "    raised = False\n"
        "except (KeyError, LookupError):\n"
        "    raised = True\n"
        "assert raised, 'get_back_func on missing (fn, argnum) should raise'"
    ),
    solution_body=(
        "class BackwardFuncLookup:\n"
        "    def __init__(self):\n"
        "        # Dict keyed by (fwd_fn, argnum). Tuples hash fine since\n"
        "        # functions are hashable by identity.\n"
        "        self._table = {}\n"
        "\n"
        "    def add_back_func(self, fwd_fn, argnum, back_fn):\n"
        "        self._table[(fwd_fn, argnum)] = back_fn\n"
        "\n"
        "    def get_back_func(self, fwd_fn, argnum):\n"
        "        # KeyError naturally propagates on miss — that's the right\n"
        "        # failure mode (a Recipe references an unwrapped op).\n"
        "        return self._table[(fwd_fn, argnum)]\n"
        "\n"
        "\n"
        "def log_back(grad_out: Tensor, out: Tensor, x: Tensor) -> Tensor:\n"
        "    return grad_out / x\n"
        "\n"
        "\n"
        "def register_log(BACK_FUNCS: 'BackwardFuncLookup') -> None:\n"
        "    BACK_FUNCS.add_back_func(t.log, 0, log_back)"
    ),
    solution_notes=(
        "**Why a tuple key and not nested dicts.** `{(fwd, argnum): fn}` is a "
        "single hash lookup. `{fwd: {argnum: fn}}` is two lookups and an extra "
        "missing-key branch. ARENA uses the flat tuple form everywhere — it's "
        "the same data model as PyTorch's C++ Node::next_edges, and pickle/repr "
        "are nicer.\n\n"
        "**Why KeyError on miss is correct.** If backprop encounters a Recipe "
        "whose forward fn was never wrapped/registered, that's a build-time bug "
        "(someone forgot a `wrap_forward_fn` call). Raising loud beats returning "
        "`None` and exploding ten frames deep with `'NoneType' is not callable`."
    ),
)

SPEC_RBF_2 = _spec(
    atom_id="register-back-fn-after-wrap",
    subtopic="Backprop: register back fn",
    recap=RECAP_REGISTER_BACK_FN,
    ex_idx=2,
    ex_title="register a binary op at TWO argnums and dispatch both",
    slug="register-binary-op-two-argnums",
    bloom="Apply",
    difficulty_num=3,
    keywords=["binary-op", "argnum", "multiply-back", "argnum-dispatch"],
    kcs=["register-back-fn-after-wrap", "arg-position-back-functions"],
    lo=(
        "Apply the (fwd_fn, argnum) registration pattern to a BINARY op by "
        "registering two back fns — one per argument position — and "
        "dispatching each independently."
    ),
    prompt_body=(
        "Implement `register_multiply(BACK_FUNCS)`. Given a fresh lookup "
        "table, register backward fns for `torch.multiply` at BOTH argnum=0 "
        "and argnum=1.\n\n"
        "The backward fns are given to you:\n"
        "- `multiply_back0(grad_out, out, x, y) -> grad_out * y`  (∂(xy)/∂x = y)\n"
        "- `multiply_back1(grad_out, out, x, y) -> grad_out * x`  (∂(xy)/∂y = x)\n\n"
        "**The key insight.** A binary op has TWO entries in the table — one "
        "for each input position. When backprop walks the graph for a node "
        "with `recipe.func == torch.multiply` and two parents at "
        "`recipe.parents = {0: x, 1: y}`, it dispatches:\n"
        "- `get_back_func(torch.multiply, 0)(grad_out, out, x, y)` → grad for x\n"
        "- `get_back_func(torch.multiply, 1)(grad_out, out, x, y)` → grad for y\n\n"
        "Note both back fns receive ALL the forward args (`x` and `y`) even "
        "though each only differentiates w.r.t. one. The uniform calling "
        "convention again — the back fn picks what it needs.\n\n"
        "The test then walks the dispatch loop for `multiply` over the parents "
        "dict and verifies both grads match the chain rule."
    ),
    stub=(
        "class BackwardFuncLookup:\n"
        "    def __init__(self):\n"
        "        self._table = {}\n"
        "    def add_back_func(self, fwd_fn, argnum, back_fn):\n"
        "        self._table[(fwd_fn, argnum)] = back_fn\n"
        "    def get_back_func(self, fwd_fn, argnum):\n"
        "        return self._table[(fwd_fn, argnum)]\n"
        "\n"
        "\n"
        "def multiply_back0(grad_out, out, x, y):\n"
        "    return grad_out * y\n"
        "\n"
        "\n"
        "def multiply_back1(grad_out, out, x, y):\n"
        "    return grad_out * x\n"
        "\n"
        "\n"
        "def register_multiply(BACK_FUNCS: BackwardFuncLookup) -> None:\n"
        '    """Register multiply_back0 at argnum=0 and multiply_back1 at argnum=1."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "BACK_FUNCS = BackwardFuncLookup()\n"
        "register_multiply(BACK_FUNCS)\n"
        "\n"
        "# Both argnums must be populated and point to the right fn.\n"
        "assert BACK_FUNCS.get_back_func(t.multiply, 0) is multiply_back0\n"
        "assert BACK_FUNCS.get_back_func(t.multiply, 1) is multiply_back1\n"
        "\n"
        "# --- dispatch loop simulating backprop for one multiply node ---\n"
        "x = t.tensor([2.0, 3.0, 4.0])\n"
        "y = t.tensor([5.0, 7.0, 11.0])\n"
        "out = x * y  # (10, 21, 44)\n"
        "grad_out = t.ones(3)\n"
        "\n"
        "# Pretend Recipe.parents = {0: x, 1: y}. Loop over (argnum, parent).\n"
        "parents = {0: x, 1: y}\n"
        "fwd_fn = t.multiply\n"
        "fwd_args = (x, y)\n"
        "grads = {}\n"
        "for argnum, parent in parents.items():\n"
        "    back_fn = BACK_FUNCS.get_back_func(fwd_fn, argnum)\n"
        "    grads[id(parent)] = back_fn(grad_out, out, *fwd_args)\n"
        "\n"
        "# d(x*y)/dx = y → grad_x should equal y.\n"
        "assert t.allclose(grads[id(x)], y), f'grad_x: {grads[id(x)]}'\n"
        "# d(x*y)/dy = x → grad_y should equal x.\n"
        "assert t.allclose(grads[id(y)], x), f'grad_y: {grads[id(y)]}'\n"
        "\n"
        "# Non-unit grad_out — chain rule should scale both grads.\n"
        "grad_out2 = t.tensor([2.0, 0.5, -1.0])\n"
        "g0 = BACK_FUNCS.get_back_func(t.multiply, 0)(grad_out2, out, x, y)\n"
        "g1 = BACK_FUNCS.get_back_func(t.multiply, 1)(grad_out2, out, x, y)\n"
        "assert t.allclose(g0, grad_out2 * y), f'g0: {g0}'\n"
        "assert t.allclose(g1, grad_out2 * x), f'g1: {g1}'\n"
        "\n"
        "# Witness vs autograd on z = (x*y).sum().\n"
        "x_ref = t.tensor([2.0, 3.0, 4.0], requires_grad=True)\n"
        "y_ref = t.tensor([5.0, 7.0, 11.0], requires_grad=True)\n"
        "(x_ref * y_ref).sum().backward()\n"
        "assert t.allclose(g0 if False else BACK_FUNCS.get_back_func(t.multiply, 0)("
        "t.ones(3), x_ref.detach() * y_ref.detach(), x_ref.detach(), y_ref.detach()), x_ref.grad)\n"
        "assert t.allclose(BACK_FUNCS.get_back_func(t.multiply, 1)("
        "t.ones(3), x_ref.detach() * y_ref.detach(), x_ref.detach(), y_ref.detach()), y_ref.grad)"
    ),
    solution_body=(
        "class BackwardFuncLookup:\n"
        "    def __init__(self):\n"
        "        self._table = {}\n"
        "    def add_back_func(self, fwd_fn, argnum, back_fn):\n"
        "        self._table[(fwd_fn, argnum)] = back_fn\n"
        "    def get_back_func(self, fwd_fn, argnum):\n"
        "        return self._table[(fwd_fn, argnum)]\n"
        "\n"
        "\n"
        "def multiply_back0(grad_out, out, x, y):\n"
        "    return grad_out * y\n"
        "\n"
        "\n"
        "def multiply_back1(grad_out, out, x, y):\n"
        "    return grad_out * x\n"
        "\n"
        "\n"
        "def register_multiply(BACK_FUNCS: BackwardFuncLookup) -> None:\n"
        "    # Two entries — one per argument position.\n"
        "    BACK_FUNCS.add_back_func(t.multiply, 0, multiply_back0)\n"
        "    BACK_FUNCS.add_back_func(t.multiply, 1, multiply_back1)"
    ),
    solution_notes=(
        "**Why N entries for an N-ary op.** PyTorch's Function class has a "
        "single `backward` that returns a tuple of grads (one per input). "
        "ARENA's BACK_FUNCS table flattens that into N independent dict entries. "
        "Same information, different shape. The dict form falls out naturally "
        "from iterating `recipe.parents.items()` and dispatching per argnum.\n\n"
        "**Why both back fns receive both args.** `multiply_back0` could be "
        "written `def multiply_back0(grad_out, out, x, y): return grad_out * y` "
        "or even `def multiply_back0(grad_out, out, _x, y): return grad_out * y`. "
        "The uniform `(grad_out, out, *original_args)` shape lets the dispatcher "
        "always call `back_fn(grad_out, out, *recipe.args, **recipe.kwargs)` — no "
        "per-fn unpacking logic."
    ),
)


# =========================================================================
# atom: wrap-forward-fn-generic  (2 exercises)
# =========================================================================

SPEC_WFF_1 = _spec(
    atom_id="wrap-forward-fn-generic",
    subtopic="Backprop: wrap forward fn",
    recap=RECAP_WRAP_FORWARD_FN,
    ex_idx=1,
    ex_title="write the wrap_forward_fn shell — unbox, call, box",
    slug="write-wrap-forward-fn-shell",
    bloom="Create",
    difficulty_num=3,
    keywords=["closure", "factory", "unbox", "box", "tensor-wrapper"],
    kcs=["wrap-forward-fn-generic", "unbox-args-tensor-to-array"],
    lo=(
        "Create the wrap_forward_fn factory closure that converts a raw "
        "numerical fn into a Tensor-aware wrapper via the unbox → call → box "
        "pattern."
    ),
    prompt_body=(
        "You are given a minimal `Tensor` wrapper class (a thin shell over a "
        "raw `torch.Tensor` stored on `.array`).\n\n"
        "Implement `wrap_forward_fn(fwd_fn)`. It must return a NEW function "
        "`tensor_func(*args, **kwargs)` that:\n\n"
        "1. **Unbox** — for each arg, if it is a `Tensor` instance, replace it "
        "   with `.array`; otherwise pass it through unchanged. (Plain ints / "
        "   floats / raw torch tensors should pass through.)\n"
        "2. **Call** — invoke `fwd_fn(*raw_args, **kwargs)`.\n"
        "3. **Box** — wrap the result in a new `Tensor(out_raw)` and return it.\n\n"
        "Do NOT touch Recipe / autograd in this drill — pure unbox/call/box. "
        "(Recipe attachment is the next drill.)\n\n"
        "The point: `wrap_forward_fn` is a closure factory. ONE definition "
        "handles every op — `log`, `exp`, `multiply`, `sum` — because the only "
        "thing that varies is the captured `fwd_fn`. Without this factory, "
        "ARENA would need 20+ hand-written wrappers."
    ),
    stub=(
        "from dataclasses import dataclass\n"
        "\n"
        "\n"
        "class Tensor:\n"
        '    """Tiny wrapper around a raw torch.Tensor stored on .array."""\n'
        "    def __init__(self, array):\n"
        "        # Coerce to torch.Tensor if a list / number snuck in.\n"
        "        self.array = array if isinstance(array, t.Tensor) else t.tensor(array)\n"
        "    def __repr__(self):\n"
        "        return f'Tensor({self.array.tolist()})'\n"
        "\n"
        "\n"
        "def wrap_forward_fn(fwd_fn):\n"
        '    """Return a Tensor-aware wrapper that unboxes args, calls fwd_fn, boxes the result."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# Wrap torch.log — unary op on Tensor.\n"
        "tlog = wrap_forward_fn(t.log)\n"
        "a = Tensor(t.tensor([1.0, t.e, t.e ** 2]))\n"
        "b = tlog(a)\n"
        "assert isinstance(b, Tensor), f'wrap must return a Tensor, got {type(b)}'\n"
        "assert isinstance(b.array, t.Tensor), '.array must be a torch.Tensor'\n"
        "expected = t.tensor([0.0, 1.0, 2.0])\n"
        "assert t.allclose(b.array, expected, atol=1e-5), f'log fail: {b.array}'\n"
        "\n"
        "# Wrap torch.multiply — binary op. Both args wrapped as Tensor.\n"
        "tmul = wrap_forward_fn(t.multiply)\n"
        "x = Tensor(t.tensor([2.0, 3.0]))\n"
        "y = Tensor(t.tensor([5.0, 7.0]))\n"
        "z = tmul(x, y)\n"
        "assert isinstance(z, Tensor)\n"
        "assert t.allclose(z.array, t.tensor([10.0, 21.0]))\n"
        "\n"
        "# Mixed: one Tensor arg, one raw scalar — scalar must pass through unchanged.\n"
        "z2 = tmul(x, 2.0)  # 2.0 is NOT a Tensor; should pass through to torch.multiply\n"
        "assert isinstance(z2, Tensor)\n"
        "assert t.allclose(z2.array, t.tensor([4.0, 6.0])), f'mixed scalar fail: {z2.array}'\n"
        "\n"
        "# kwargs must thread through to fwd_fn.\n"
        "tsum = wrap_forward_fn(t.sum)\n"
        "m = Tensor(t.tensor([[1.0, 2.0], [3.0, 4.0]]))\n"
        "row_sums = tsum(m, dim=1)\n"
        "assert t.allclose(row_sums.array, t.tensor([3.0, 7.0])), f'kwargs fail: {row_sums.array}'\n"
        "\n"
        "# wrap_forward_fn must produce a DIFFERENT callable each call (a new closure),\n"
        "# and the captured fwd_fn must NOT leak between wrapped fns.\n"
        "tneg = wrap_forward_fn(t.neg)\n"
        "assert tneg is not tlog, 'each wrap must return a fresh closure'\n"
        "n = tneg(Tensor(t.tensor([1.0, -2.0])))\n"
        "assert t.allclose(n.array, t.tensor([-1.0, 2.0])), 'neg wrapper fail'\n"
        "# tlog still works after wrapping tneg.\n"
        "assert t.allclose(tlog(Tensor(t.tensor([1.0]))).array, t.tensor([0.0]))"
    ),
    solution_body=(
        "from dataclasses import dataclass\n"
        "\n"
        "\n"
        "class Tensor:\n"
        "    def __init__(self, array):\n"
        "        self.array = array if isinstance(array, t.Tensor) else t.tensor(array)\n"
        "    def __repr__(self):\n"
        "        return f'Tensor({self.array.tolist()})'\n"
        "\n"
        "\n"
        "def wrap_forward_fn(fwd_fn):\n"
        "    def tensor_func(*args, **kwargs):\n"
        "        # 1. Unbox: Tensor → .array; anything else passes through.\n"
        "        raw_args = tuple(a.array if isinstance(a, Tensor) else a for a in args)\n"
        "        # 2. Call the wrapped fn on raw tensors.\n"
        "        out_raw = fwd_fn(*raw_args, **kwargs)\n"
        "        # 3. Box the result back into a Tensor.\n"
        "        return Tensor(out_raw)\n"
        "    return tensor_func"
    ),
    solution_notes=(
        "**Why a closure beats a class.** Each wrapped op gets its own "
        "`tensor_func` closure that captures `fwd_fn`. No shared mutable state. "
        "The factory pattern is dense — three lines of body produce N callables.\n\n"
        "**Why `isinstance(a, Tensor) else a`.** Real call sites mix wrapped "
        "Tensors and raw scalars (`a * 2.0`, `x.sum(dim=1)`). The unbox map "
        "must be a no-op on non-Tensor args so the wrapped fn receives them "
        "unchanged. The dual side (boxing scalars TO Tensor) is the "
        "`__rmul__` problem and lives elsewhere.\n\n"
        "**This drill is intentionally Recipe-free.** Adding the Recipe is one "
        "extra block — see the next exercise. The mental model is cleaner if "
        "you nail the unbox/call/box mechanism first."
    ),
)

SPEC_WFF_2 = _spec(
    atom_id="wrap-forward-fn-generic",
    subtopic="Backprop: wrap forward fn",
    recap=RECAP_WRAP_FORWARD_FN,
    ex_idx=2,
    ex_title="extend wrap_forward_fn with kwargs pass-through and is_differentiable",
    slug="wrap-forward-fn-with-kwargs-and-is-differentiable",
    bloom="Create",
    difficulty_num=4,
    keywords=["kwargs", "is-differentiable", "recipe", "closure"],
    kcs=["wrap-forward-fn-generic", "is-differentiable-flag"],
    lo=(
        "Create an extended wrap_forward_fn that threads kwargs through to "
        "both the forward call and the Recipe, and respects an is_differentiable "
        "flag to short-circuit Recipe construction for non-diff ops."
    ),
    prompt_body=(
        "Extend the simple `wrap_forward_fn` from the previous drill into the "
        "full ARENA version. The minimal `Tensor`, `Recipe`, and a "
        "`requires_grad` flag are scaffolded for you.\n\n"
        "Implement `wrap_forward_fn(fwd_fn, is_differentiable=True)` so that "
        "the returned `tensor_func(*args, **kwargs)`:\n\n"
        "1. **Unbox + call** as before — Tensors → `.array`, then "
        "   `fwd_fn(*raw_args, **kwargs)` (kwargs threaded through).\n"
        "2. Compute `requires_grad = is_differentiable AND any(input is Tensor "
        "   with requires_grad=True)`. (No global toggle in this drill — "
        "   simpler than ARENA's three-conjunct version.)\n"
        "3. **Box** the result: `out = Tensor(out_raw, requires_grad)`.\n"
        "4. If `requires_grad` is True, **attach a Recipe**: "
        "   `out.recipe = Recipe(fwd_fn, raw_args, kwargs, parents)` where "
        "   `parents = {i: a for i, a in enumerate(args) if isinstance(a, Tensor)}`. "
        "   Note `kwargs` lands in the Recipe verbatim — the reverse pass "
        "   needs them too (e.g. `dim=` for `sum_back`).\n"
        "5. Otherwise leave `out.recipe = None`.\n\n"
        "**The two new pieces** beyond drill 1:\n"
        "- **kwargs pass-through to Recipe** — ARENA's `sum_back` needs to "
        "  know the `dim=` and `keepdim=` that the forward `sum` was called "
        "  with; the Recipe carries them.\n"
        "- **`is_differentiable=False`** — for ops like `torch.eq` that return "
        "  bools, we wrap them so they accept Tensors, but we skip Recipe "
        "  construction (no backward pass possible)."
    ),
    stub=(
        "from dataclasses import dataclass, field\n"
        "\n"
        "\n"
        "@dataclass\n"
        "class Recipe:\n"
        "    func: object\n"
        "    args: tuple\n"
        "    kwargs: dict\n"
        "    parents: dict\n"
        "\n"
        "\n"
        "class Tensor:\n"
        "    def __init__(self, array, requires_grad: bool = False):\n"
        "        self.array = array if isinstance(array, t.Tensor) else t.tensor(array)\n"
        "        self.requires_grad = requires_grad\n"
        "        self.recipe = None\n"
        "    def __repr__(self):\n"
        "        return f'Tensor({self.array.tolist()}, requires_grad={self.requires_grad})'\n"
        "\n"
        "\n"
        "def wrap_forward_fn(fwd_fn, is_differentiable: bool = True):\n"
        '    """Tensor-aware wrapper with Recipe attachment and is_differentiable gating."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- baseline: wrap torch.log, requires_grad propagates ---\n"
        "tlog = wrap_forward_fn(t.log)\n"
        "a = Tensor(t.tensor([1.0, t.e]), requires_grad=True)\n"
        "b = tlog(a)\n"
        "assert b.requires_grad, 'requires_grad must propagate from input'\n"
        "assert b.recipe is not None, 'Recipe must be attached when requires_grad=True'\n"
        "assert b.recipe.func is t.log\n"
        "assert b.recipe.parents == {0: a}, f'parents wrong: {b.recipe.parents}'\n"
        "assert b.recipe.kwargs == {}, f'kwargs wrong: {b.recipe.kwargs}'\n"
        "assert t.allclose(b.array, t.tensor([0.0, 1.0]), atol=1e-5)\n"
        "\n"
        "# --- no requires_grad anywhere → no Recipe, no requires_grad ---\n"
        "a_off = Tensor(t.tensor([1.0, t.e]), requires_grad=False)\n"
        "b_off = tlog(a_off)\n"
        "assert b_off.requires_grad is False, 'no input wants grad → output mustnt either'\n"
        "assert b_off.recipe is None, 'no Recipe when requires_grad=False'\n"
        "\n"
        "# --- binary op with one tracked input + one raw scalar ---\n"
        "tmul = wrap_forward_fn(t.multiply)\n"
        "x = Tensor(t.tensor([2.0, 3.0]), requires_grad=True)\n"
        "z = tmul(x, 5.0)\n"
        "assert z.requires_grad\n"
        "assert z.recipe.parents == {0: x}, f'scalar must NOT be in parents: {z.recipe.parents}'\n"
        "assert z.recipe.args[1] == 5.0, 'raw scalar passes through to recipe.args'\n"
        "assert t.allclose(z.array, t.tensor([10.0, 15.0]))\n"
        "\n"
        "# --- kwargs threaded through to BOTH forward call and Recipe ---\n"
        "tsum = wrap_forward_fn(t.sum)\n"
        "m = Tensor(t.tensor([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)\n"
        "s = tsum(m, dim=1, keepdim=True)\n"
        "assert t.allclose(s.array, t.tensor([[3.0], [7.0]])), f'kwargs not used in fwd: {s.array}'\n"
        "assert s.recipe.kwargs == {'dim': 1, 'keepdim': True}, (\n"
        "    f'kwargs missing from Recipe: {s.recipe.kwargs}'\n"
        ")\n"
        "\n"
        "# --- is_differentiable=False short-circuits Recipe ---\n"
        "teq = wrap_forward_fn(t.eq, is_differentiable=False)\n"
        "p = Tensor(t.tensor([1.0, 2.0, 3.0]), requires_grad=True)\n"
        "q = Tensor(t.tensor([1.0, 5.0, 3.0]), requires_grad=True)\n"
        "r = teq(p, q)\n"
        "assert r.requires_grad is False, 'is_differentiable=False forces requires_grad=False'\n"
        "assert r.recipe is None, 'is_differentiable=False forces no Recipe'\n"
        "assert t.equal(r.array, t.tensor([True, False, True])), f'eq value wrong: {r.array}'\n"
        "\n"
        "# --- two tracked inputs → parents has both ---\n"
        "x2 = Tensor(t.tensor([2.0]), requires_grad=True)\n"
        "y2 = Tensor(t.tensor([3.0]), requires_grad=True)\n"
        "z2 = tmul(x2, y2)\n"
        "assert z2.recipe.parents == {0: x2, 1: y2}, f'two-input parents: {z2.recipe.parents}'"
    ),
    solution_body=(
        "from dataclasses import dataclass, field\n"
        "\n"
        "\n"
        "@dataclass\n"
        "class Recipe:\n"
        "    func: object\n"
        "    args: tuple\n"
        "    kwargs: dict\n"
        "    parents: dict\n"
        "\n"
        "\n"
        "class Tensor:\n"
        "    def __init__(self, array, requires_grad: bool = False):\n"
        "        self.array = array if isinstance(array, t.Tensor) else t.tensor(array)\n"
        "        self.requires_grad = requires_grad\n"
        "        self.recipe = None\n"
        "    def __repr__(self):\n"
        "        return f'Tensor({self.array.tolist()}, requires_grad={self.requires_grad})'\n"
        "\n"
        "\n"
        "def wrap_forward_fn(fwd_fn, is_differentiable: bool = True):\n"
        "    def tensor_func(*args, **kwargs):\n"
        "        # 1. Unbox.\n"
        "        raw_args = tuple(a.array if isinstance(a, Tensor) else a for a in args)\n"
        "        # 2. Call (kwargs thread through).\n"
        "        out_raw = fwd_fn(*raw_args, **kwargs)\n"
        "        # 3. Compute requires_grad — both gates.\n"
        "        requires_grad = is_differentiable and any(\n"
        "            isinstance(a, Tensor) and a.requires_grad for a in args\n"
        "        )\n"
        "        # 4. Box.\n"
        "        out = Tensor(out_raw, requires_grad)\n"
        "        # 5. Attach Recipe if tracked. parents = dict-by-argidx filtered to Tensors.\n"
        "        if requires_grad:\n"
        "            parents = {i: a for i, a in enumerate(args) if isinstance(a, Tensor)}\n"
        "            out.recipe = Recipe(fwd_fn, raw_args, kwargs, parents)\n"
        "        return out\n"
        "    return tensor_func"
    ),
    solution_notes=(
        "**Why `parents` filters non-Tensors.** A `Tensor * 5.0` call has "
        "`args = (Tensor, 5.0)`. We can't store `5.0` in `parents` because the "
        "reverse pass iterates `parents.items()` and dispatches a back fn per "
        "parent — there's no gradient to compute for a raw scalar. The "
        "dict-by-argidx-with-Tensor-filter pattern keeps the keys aligned with "
        "the original arg positions (so `BACK_FUNCS.get_back_func(fn, argnum)` "
        "uses the right argnum).\n\n"
        "**Why kwargs land verbatim in Recipe.** `sum_back(grad_out, out, x, "
        "dim=..., keepdim=...)` needs the same `dim`/`keepdim` the forward "
        "used, or the broadcast-shape inverse comes out wrong. ARENA's reverse "
        "pass does `back_fn(grad_out, node.array, *node.recipe.args, "
        "**node.recipe.kwargs)` — kwargs flow back through, same shape.\n\n"
        "**Why `is_differentiable` short-circuits before the Recipe.** `torch.eq` "
        "returns bools — there is no meaningful gradient. We still want to wrap "
        "it so users can write `Tensor(...) == Tensor(...)`, but `requires_grad` "
        "must be False on the output (and no Recipe → no wasted memory)."
    ),
)


# =========================================================================
# atom: param-grad-access  (1 exercise)
# =========================================================================

SPEC_PG_1 = _spec(
    atom_id="param-grad-access",
    subtopic="PyTorch: param.grad access",
    recap=RECAP_PARAM_GRAD,
    ex_idx=1,
    ex_title="iterate model.parameters() and read .grad with the None guard",
    slug="iterate-parameters-read-grad-with-none-guard",
    bloom="Apply",
    difficulty_num=2,
    keywords=["parameters", "grad", "iterate", "none-guard", "sgd-step"],
    kcs=["param-grad-access", "zero-grad-set-none"],
    lo=(
        "Apply the parameters() → p.grad pattern (with None-guard) to compute "
        "an SGD step manually across a multi-parameter module."
    ),
    prompt_body=(
        "Implement `manual_sgd_step(params, lr)`. Given an iterable of "
        "tensors that already have `.grad` populated (or `.grad is None`), "
        "apply the canonical SGD step to each.\n\n"
        "**For each parameter p:**\n"
        "1. If `p.grad is None`, **skip** it (the standard guard — a parameter "
        "   that didn't participate in any forward pass has no gradient yet, "
        "   and you can't subtract `None` from a tensor).\n"
        "2. Otherwise update **in place**: `p.data -= lr * p.grad`. We use "
        "   `.data` to bypass autograd tracking (we don't want the SGD step "
        "   itself to build a Recipe).\n\n"
        "**Then implement `zero_grads(params)`** — for each parameter, set "
        "`p.grad = None`. This is the PyTorch-recommended zero strategy "
        "(memory-cheaper than `p.grad.zero_()` because the gradient tensor "
        "becomes garbage-collectable, and matches `optimizer.zero_grad("
        "set_to_none=True)`).\n\n"
        "Inputs are plain `nn.Parameter` instances (a Tensor subclass). The "
        "test populates `.grad` by hand (no real backward pass needed) and "
        "checks that updates match `p_new = p_old - lr * grad` elementwise."
    ),
    stub=(
        "def manual_sgd_step(params, lr: float) -> None:\n"
        '    """In-place SGD update: p.data -= lr * p.grad, skipping None grads."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def zero_grads(params) -> None:\n"
        '    """Set p.grad = None for every param (PyTorch-style zero_grad)."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "import torch.nn as nn\n"
        "\n"
        "# Hand-roll a tiny module with 3 parameters of different shapes.\n"
        "w = nn.Parameter(t.tensor([[1.0, 2.0], [3.0, 4.0]]))\n"
        "b = nn.Parameter(t.tensor([0.5, -0.5]))\n"
        "scale = nn.Parameter(t.tensor(2.0))  # 0-dim\n"
        "params = [w, b, scale]\n"
        "\n"
        "# Hand-populate gradients (would normally come from .backward()).\n"
        "w.grad = t.tensor([[0.1, 0.2], [0.3, 0.4]])\n"
        "b.grad = t.tensor([1.0, 2.0])\n"
        "scale.grad = t.tensor(10.0)\n"
        "\n"
        "lr = 0.1\n"
        "# Snapshot the *expected* post-step values BEFORE the in-place update.\n"
        "expected_w = w.data - lr * w.grad\n"
        "expected_b = b.data - lr * b.grad\n"
        "expected_scale = scale.data - lr * scale.grad\n"
        "\n"
        "manual_sgd_step(params, lr)\n"
        "\n"
        "assert t.allclose(w.data, expected_w), f'w fail: {w.data}'\n"
        "assert t.allclose(b.data, expected_b), f'b fail: {b.data}'\n"
        "assert t.allclose(scale.data, expected_scale), f'scale fail: {scale.data}'\n"
        "\n"
        "# Critical: the update was in-place — identity preserved.\n"
        "assert isinstance(w, nn.Parameter), 'in-place update must preserve Parameter type'\n"
        "\n"
        "# --- None-grad guard ---\n"
        "fresh = nn.Parameter(t.tensor([10.0, 20.0]))  # .grad starts as None\n"
        "assert fresh.grad is None, 'fresh parameter must start with .grad = None'\n"
        "before = fresh.data.clone()\n"
        "manual_sgd_step([fresh], lr=0.1)  # must NOT raise, must NOT change fresh.data\n"
        "assert t.allclose(fresh.data, before), (\n"
        "    f'param with None grad must be skipped: was {before}, now {fresh.data}'\n"
        ")\n"
        "\n"
        "# Mixed: some params with grad, some None — the None ones get skipped.\n"
        "with_grad = nn.Parameter(t.tensor([1.0]))\n"
        "with_grad.grad = t.tensor([0.5])\n"
        "without_grad = nn.Parameter(t.tensor([100.0]))  # .grad None\n"
        "expected_wg = with_grad.data - 0.5 * 0.5  # lr=0.5, grad=0.5\n"
        "manual_sgd_step([with_grad, without_grad], lr=0.5)\n"
        "assert t.allclose(with_grad.data, expected_wg)\n"
        "assert t.allclose(without_grad.data, t.tensor([100.0])), 'mixed-None param drifted'\n"
        "\n"
        "# --- zero_grads ---\n"
        "zero_grads(params)\n"
        "for p in params:\n"
        "    assert p.grad is None, f'zero_grads must set .grad to None, got {p.grad}'\n"
        "\n"
        "# Step after zero_grads with no new backward must be a no-op (all grads None).\n"
        "snapshot = [p.data.clone() for p in params]\n"
        "manual_sgd_step(params, lr=999.0)  # huge lr, but every grad is None → no-op\n"
        "for p, snap in zip(params, snapshot):\n"
        "    assert t.allclose(p.data, snap), 'post-zero step must not move params'"
    ),
    solution_body=(
        "def manual_sgd_step(params, lr: float) -> None:\n"
        "    for p in params:\n"
        "        if p.grad is None:\n"
        "            # Param didn't see any forward pass since last zero_grad.\n"
        "            # Subtracting None from a tensor raises TypeError, so skip.\n"
        "            continue\n"
        "        # In-place: .data bypasses autograd tracking for the update itself.\n"
        "        p.data -= lr * p.grad\n"
        "\n"
        "\n"
        "def zero_grads(params) -> None:\n"
        "    for p in params:\n"
        "        # set_to_none=True style — cheaper than p.grad.zero_().\n"
        "        # The grad tensor becomes garbage-collectable.\n"
        "        p.grad = None"
    ),
    solution_notes=(
        "**Why guard on `p.grad is None`.** In PyTorch, fresh `Parameter`s have "
        "`.grad = None` until the first `backward()` populates them. After "
        "`zero_grad(set_to_none=True)`, `.grad` becomes `None` again. If you "
        "blindly do `p.data -= lr * p.grad` you'll get `TypeError: unsupported "
        "operand type(s) for *: 'float' and 'NoneType'`.\n\n"
        "**Why `p.data -=` and not `p -=`.** `p -= lr * p.grad` would trigger "
        "autograd because `p.requires_grad` is True (Parameters always do). "
        "`.data` is the raw underlying tensor — writing to it doesn't build a "
        "Recipe. The same reason ARENA wraps the SGD step in `NoGrad()`.\n\n"
        "**Why `p.grad = None` over `p.grad.zero_()`.** Two reasons:\n"
        "- Memory: the gradient tensor becomes garbage-collectable, freeing up "
        "  GPU memory between forward passes. `zero_()` keeps the allocation "
        "  alive.\n"
        "- Semantics: `None` means \"no gradient computed yet\" (a clearer "
        "  invariant than \"the zero tensor pretending to be no gradient\"). "
        "  This is why PyTorch made `set_to_none=True` the default in 1.7."
    ),
)


# =========================================================================
# atom: buffer-copy_-inplace  (1 exercise)
# =========================================================================

SPEC_BC_1 = _spec(
    atom_id="buffer-copy_-inplace",
    subtopic="PyTorch: in-place buffer copy",
    recap=RECAP_BUFFER_COPY,
    ex_idx=1,
    ex_title="update a BatchNorm running_mean buffer with copy_",
    slug="update-running-mean-buffer-with-copy",
    bloom="Apply",
    difficulty_num=2,
    keywords=["copy_", "buffer", "in-place", "batchnorm", "running-mean", "identity"],
    kcs=["buffer-copy_-inplace", "inplace-param-update"],
    lo=(
        "Apply `tensor.copy_(other)` to update a running-mean buffer in place "
        "while preserving the buffer's storage identity (so registered-buffer "
        "links survive)."
    ),
    prompt_body=(
        "Implement `update_running_mean(running_mean, batch_mean, momentum)`. "
        "This is the BatchNorm running-stats update, simplified to a single "
        "buffer.\n\n"
        "**The math.** Exponential moving average:\n"
        "`new_running = (1 - momentum) * running_mean + momentum * batch_mean`\n"
        "(PyTorch's BatchNorm uses this exact form when `momentum=0.1`.)\n\n"
        "**The critical mechanic.** You must write the new value into the "
        "EXISTING `running_mean` tensor in place using `.copy_()`. Do NOT:\n"
        "- Reassign: `running_mean = ...` — the caller's reference is "
        "  unaffected (the buffer registration is lost in real nn.Module code).\n"
        "- Use `*=` or `+=` separately and accumulate — pointless extra storage.\n"
        "- Use `.data =` — works but bypasses `copy_`'s shape/dtype checks.\n\n"
        "Use exactly: `running_mean.copy_(new_value)`. Return nothing (the "
        "function mutates `running_mean` in place).\n\n"
        "The test asserts that `id(running_mean)` is preserved before/after "
        "the call — proof the buffer identity survived."
    ),
    stub=(
        "def update_running_mean(running_mean: Tensor, batch_mean: Tensor, momentum: float) -> None:\n"
        '    """In-place EMA update of running_mean. Mutates running_mean, returns None."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "import torch.nn as nn\n"
        "\n"
        "# Simulate a registered buffer (like nn.BatchNorm1d's running_mean).\n"
        "class TinyBN(nn.Module):\n"
        "    def __init__(self, n):\n"
        "        super().__init__()\n"
        "        self.register_buffer('running_mean', t.zeros(n))\n"
        "\n"
        "bn = TinyBN(4)\n"
        "original_id = id(bn.running_mean)\n"
        "original_storage = bn.running_mean.untyped_storage().data_ptr()\n"
        "\n"
        "# First update — running_mean starts at zeros, batch_mean at ones, momentum 0.1.\n"
        "batch_mean = t.tensor([1.0, 1.0, 1.0, 1.0])\n"
        "update_running_mean(bn.running_mean, batch_mean, momentum=0.1)\n"
        "\n"
        "expected = t.tensor([0.1, 0.1, 0.1, 0.1])  # 0.9*0 + 0.1*1\n"
        "assert t.allclose(bn.running_mean, expected), f'first update: {bn.running_mean}'\n"
        "\n"
        "# CRITICAL: identity preserved → buffer registration intact.\n"
        "assert id(bn.running_mean) == original_id, (\n"
        "    f'tensor object identity broken — buffer no longer the same object'\n"
        ")\n"
        "assert bn.running_mean.untyped_storage().data_ptr() == original_storage, (\n"
        "    'underlying storage was reallocated — copy_ wasnt in-place'\n"
        ")\n"
        "\n"
        "# Second update — EMA should keep decaying toward batch_mean.\n"
        "update_running_mean(bn.running_mean, batch_mean, momentum=0.1)\n"
        "# 0.9 * 0.1 + 0.1 * 1.0 = 0.19\n"
        "expected2 = t.tensor([0.19, 0.19, 0.19, 0.19])\n"
        "assert t.allclose(bn.running_mean, expected2, atol=1e-5), (\n"
        "    f'second update: {bn.running_mean}, expected {expected2}'\n"
        ")\n"
        "\n"
        "# Identity STILL preserved after many updates.\n"
        "for _ in range(20):\n"
        "    update_running_mean(bn.running_mean, batch_mean, momentum=0.1)\n"
        "assert id(bn.running_mean) == original_id, 'identity must survive many updates'\n"
        "# After many steps the EMA should be very close to the constant batch_mean.\n"
        "assert t.allclose(bn.running_mean, batch_mean, atol=0.2), (\n"
        "    f'EMA didnt converge: {bn.running_mean}'\n"
        ")\n"
        "\n"
        "# --- state_dict check: buffer must still appear ---\n"
        "sd = bn.state_dict()\n"
        "assert 'running_mean' in sd, 'buffer dropped from state_dict — identity break'\n"
        "assert t.allclose(sd['running_mean'], bn.running_mean)\n"
        "\n"
        "# --- shape mismatch must raise ---\n"
        "buf = t.zeros(3)\n"
        "wrong = t.tensor([1.0, 2.0])  # length-2 vs length-3 buffer\n"
        "try:\n"
        "    update_running_mean(buf, wrong, momentum=0.1)\n"
        "    raised = False\n"
        "except RuntimeError:\n"
        "    raised = True\n"
        "assert raised, 'shape mismatch should raise (copy_ enforces shape)'\n"
        "\n"
        "# --- different momentum values ---\n"
        "buf = t.zeros(2)\n"
        "update_running_mean(buf, t.tensor([10.0, 20.0]), momentum=1.0)\n"
        "# momentum=1 → fully overwrite with batch_mean.\n"
        "assert t.allclose(buf, t.tensor([10.0, 20.0])), f'momentum=1 fail: {buf}'\n"
        "\n"
        "buf = t.tensor([5.0, 5.0])\n"
        "update_running_mean(buf, t.tensor([99.0, 99.0]), momentum=0.0)\n"
        "# momentum=0 → freeze (running_mean unchanged).\n"
        "assert t.allclose(buf, t.tensor([5.0, 5.0])), f'momentum=0 fail: {buf}'"
    ),
    solution_body=(
        "def update_running_mean(running_mean: Tensor, batch_mean: Tensor, momentum: float) -> None:\n"
        "    # Compute the new EMA value in a fresh tensor, then write it into\n"
        "    # the existing running_mean's storage via copy_. This preserves\n"
        "    # id(running_mean) — critical for nn.Module's registered-buffer link.\n"
        "    new_value = (1 - momentum) * running_mean + momentum * batch_mean\n"
        "    running_mean.copy_(new_value)"
    ),
    solution_notes=(
        "**Why `copy_` and not `=`.** In `nn.Module.register_buffer('name', "
        "tensor)`, PyTorch stores the tensor reference in `self._buffers["
        "'name']` AND aliases `self.name`. If you do `self.name = new_tensor`, "
        "you rebind the attribute but `_buffers['name']` still points to the "
        "OLD tensor. `.to(device)`, `.state_dict()`, `.load_state_dict()` all "
        "walk `_buffers` — they'll see stale data. `copy_` writes into the "
        "old tensor's storage, so both references see the update.\n\n"
        "**Why `copy_` over `.data =`.** `running_mean.data = new_value` does "
        "work in PyTorch (and preserves identity) but skips `copy_`'s "
        "shape/dtype checks, masking bugs. PyTorch's own BatchNorm uses "
        "`copy_` (or `*=` / `+=` directly on running_mean) — never `.data =`.\n\n"
        "**The EMA math.** When `momentum=0.1`, the running mean has effective "
        "half-life ≈ 7 batches. PyTorch chose 0.1 as the default for BatchNorm "
        "because it's slow enough to smooth across batches but fast enough to "
        "track training-time distribution shift."
    ),
)


# ---------------------------------------------------------------- emit

ALL_SPECS = [
    SPEC_BFS_1,
    SPEC_BFS_2,
    SPEC_RBF_1,
    SPEC_RBF_2,
    SPEC_WFF_1,
    SPEC_WFF_2,
    SPEC_PG_1,
    SPEC_BC_1,
]


for spec in ALL_SPECS:
    path = emit_standalone(spec)
    rel = path.relative_to(path.parents[3])
    print(f"wrote {rel}")
