#!/usr/bin/env python3
"""Author 6 COMPOSITE procedural drills (batch-15, L assignments).

Each composite exercises 2-3 atoms together in a SINGLE function whose
solution genuinely uses every atom (not just mentions them).

  cx1 — wrap-forward-fn-generic + register-back-fn-after-wrap + backward-fn-signature
        ("wire a new op into autograd" trifecta — log demo)
  cx2 — multiply-back + unbroadcast-pattern + arg-position-back-functions
        (mul back-fns: per-arg chain rule + unbroadcast + argnum dispatch)
  cx3 — multiply-back + arg-position-back-functions
        (mul_back0 vs mul_back1 dispatch via argnum, no broadcasting)
  cx4 — exp-back + backward-fn-signature
        (use cached `out` instead of recomputing exp(x))
  cx5 — log-back + chain-rule-elementwise
        (one-line elementwise chain rule grad_out / x)
  cx6 — negative-back + backward-fn-signature
        (-grad_out plus the canonical (grad_out, out, x) signature contract)

NOTE: the brief listed `exp-backward-via-out` for cx4, but that atom_id does
not exist in /tmp/drill_atoms.json. The closest atom (same theme: "use cached
`out`") is `exp-back`, which is what cx4 uses. Subtopic resolves correctly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_composite import emit_composite

# ----------------------------------------------------------------- atom subtopics
INV = {a["atom_id"]: a for a in json.load(open("/tmp/drill_atoms.json"))}


def _subs(atom_ids):
    out = []
    for a in atom_ids:
        if a not in INV:
            raise KeyError(f"atom_id {a!r} not in /tmp/drill_atoms.json")
        out.append(INV[a]["subtopic"])
    return out


# ================================================================= cx1
CX1_ATOMS = [
    "wrap-forward-fn-generic",
    "register-back-fn-after-wrap",
    "backward-fn-signature",
]
spec_1 = {
    "atom_ids": CX1_ATOMS,
    "subtopics": _subs(CX1_ATOMS),
    "primary_atom": "wrap-forward-fn-generic",
    "part": "part4",
    "exercise_index": 1,
    "exercise_title": "wire a new op into autograd — wrap + register + signature",
    "slug": "wire-new-op-into-autograd",
    "atom_recap_md": (
        "## Wiring a new op into autograd — the trifecta\n"
        "\n"
        "Three atoms compose into one workflow whenever ARENA adds a new differentiable op:\n"
        "\n"
        "1. **`wrap-forward-fn-generic`** — turn a raw numerical fn (`torch.log`) into a\n"
        "   `Tensor`-aware fn by unboxing args, calling the raw fn, then boxing the result\n"
        "   inside a fresh closure.\n"
        "2. **`backward-fn-signature`** — write a back fn that follows the uniform\n"
        "   `(grad_out, out, *args) -> grad_in` contract so the dispatcher can call it\n"
        "   generically. For elementwise ops the body is `grad_out * local_derivative`.\n"
        "3. **`register-back-fn-after-wrap`** — store the back fn in a lookup table keyed\n"
        "   by `(forward_fn, argnum)` so the reverse pass can find it by tuple key.\n"
        "\n"
        "Composition: the order is fixed — wrap, then write back fn with canonical\n"
        "signature, then register under the `(fwd_fn, argnum)` key. Skipping any one\n"
        "step leaves the op invisible to the dispatcher.\n"
    ),
    "prompt_body": (
        "Wire `torch.log` end-to-end into the tiny autograd. Build a single function\n"
        "`cx1_wire_log()` that returns a 3-tuple `(tlog, BACK_FUNCS, log_back)` where:\n"
        "\n"
        "- `tlog` is the `Tensor`-aware wrapper for `torch.log`, produced via your\n"
        "  `wrap_forward_fn(t.log)`. Calling `tlog(Tensor([1., e, e**2]))` must return\n"
        "  a `Tensor` whose `.array` is `[0., 1., 2.]`.\n"
        "- `BACK_FUNCS` is a fresh `BackwardFuncLookup` with `log_back` registered at\n"
        "  `(torch.log, 0)`.\n"
        "- `log_back(grad_out, out, x)` follows the canonical back-fn signature and\n"
        "  returns `grad_out / x`.\n"
        "\n"
        "The test then exercises the trifecta: it (a) calls `tlog` on a `Tensor`,\n"
        "(b) looks up `(torch.log, 0)` from the table, and (c) invokes the looked-up\n"
        "back fn through that dispatch path.\n"
    ),
    "stub_body": (
        "class Tensor:\n"
        "    \"\"\"Thin wrapper around a raw torch.Tensor stored on .array.\"\"\"\n"
        "    def __init__(self, array):\n"
        "        self.array = array if isinstance(array, t.Tensor) else t.tensor(array)\n"
        "    def __repr__(self):\n"
        "        return f'Tensor({self.array.tolist()})'\n"
        "\n"
        "\n"
        "class BackwardFuncLookup:\n"
        "    def __init__(self):\n"
        "        raise NotImplementedError()\n"
        "    def add_back_func(self, fwd_fn, argnum, back_fn):\n"
        "        raise NotImplementedError()\n"
        "    def get_back_func(self, fwd_fn, argnum):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "\n"
        "def wrap_forward_fn(fwd_fn):\n"
        "    \"\"\"Return a Tensor-aware wrapper: unbox args, call fwd_fn, box result.\"\"\"\n"
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def cx1_wire_log():\n"
        "    \"\"\"Return (tlog, BACK_FUNCS, log_back) — all three atoms composed.\"\"\"\n"
        "    raise NotImplementedError()\n"
    ),
    "test_body": (
        "import math\n"
        "tlog, BACK_FUNCS, log_back = cx1_wire_log()\n"
        "\n"
        "# (1) wrap_forward_fn produced a Tensor-aware wrapper for torch.log\n"
        "a = Tensor(t.tensor([1.0, math.e, math.e ** 2]))\n"
        "b = tlog(a)\n"
        "assert isinstance(b, Tensor), f'tlog must return a Tensor, got {type(b)}'\n"
        "assert isinstance(b.array, t.Tensor), '.array must be a torch.Tensor'\n"
        "assert t.allclose(b.array, t.tensor([0.0, 1.0, 2.0]), atol=1e-5), f'tlog value: {b.array}'\n"
        "\n"
        "# (2) BACK_FUNCS holds log_back at the (torch.log, 0) key — dispatch path works\n"
        "fn = BACK_FUNCS.get_back_func(t.log, 0)\n"
        "assert fn is log_back, f'dispatch returned {fn}, expected the same log_back fn object'\n"
        "\n"
        "# (3) canonical signature: log_back(grad_out, out, x) = grad_out / x\n"
        "x = t.tensor([1.0, 2.0, 4.0])\n"
        "out = t.log(x)\n"
        "g = log_back(t.ones(3), out, x)\n"
        "assert g.shape == x.shape\n"
        "assert t.allclose(g, t.tensor([1.0, 0.5, 0.25])), f'log_back value: {g}'\n"
        "\n"
        "# (4) end-to-end: call the looked-up back fn through the table\n"
        "fn_again = BACK_FUNCS.get_back_func(t.log, 0)\n"
        "g2 = fn_again(t.tensor([3.0, -2.0, 10.0]), out, x)\n"
        "assert t.allclose(g2, t.tensor([3.0, -1.0, 2.5])), f'dispatched call: {g2}'\n"
    ),
    "solution_body": (
        "class Tensor:\n"
        "    def __init__(self, array):\n"
        "        self.array = array if isinstance(array, t.Tensor) else t.tensor(array)\n"
        "    def __repr__(self):\n"
        "        return f'Tensor({self.array.tolist()})'\n"
        "\n"
        "class BackwardFuncLookup:\n"
        "    def __init__(self):\n"
        "        self._table = {}\n"
        "    def add_back_func(self, fwd_fn, argnum, back_fn):\n"
        "        self._table[(fwd_fn, argnum)] = back_fn\n"
        "    def get_back_func(self, fwd_fn, argnum):\n"
        "        return self._table[(fwd_fn, argnum)]\n"
        "\n"
        "def wrap_forward_fn(fwd_fn):\n"
        "    def tensor_func(*args, **kwargs):\n"
        "        raw = tuple(a.array if isinstance(a, Tensor) else a for a in args)\n"
        "        return Tensor(fwd_fn(*raw, **kwargs))\n"
        "    return tensor_func\n"
        "\n"
        "def cx1_wire_log():\n"
        "    # atom 1: wrap the forward fn into a Tensor-aware closure\n"
        "    tlog = wrap_forward_fn(t.log)\n"
        "    # atom 2: write a back fn with the canonical (grad_out, out, x) signature\n"
        "    def log_back(grad_out, out, x):\n"
        "        return grad_out / x\n"
        "    # atom 3: register it under the (fwd_fn, argnum) key after wrapping\n"
        "    BACK_FUNCS = BackwardFuncLookup()\n"
        "    BACK_FUNCS.add_back_func(t.log, 0, log_back)\n"
        "    return tlog, BACK_FUNCS, log_back\n"
    ),
    "solution_notes": (
        "All three atoms compose in one function: `tlog = wrap_forward_fn(...)` is the\n"
        "wrap atom, the `def log_back(grad_out, out, x): return grad_out / x` line is the\n"
        "signature-contract atom, and `BACK_FUNCS.add_back_func(t.log, 0, log_back)` is the\n"
        "register atom. The test dispatches through `BACK_FUNCS.get_back_func(t.log, 0)` —\n"
        "if any one of the three steps is wrong, dispatch fails."
    ),
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["wrap-forward-fn-generic", "register-back-fn-after-wrap", "backward-fn-signature"],
    "lo": (
        "Apply the wrap-forward-fn-generic + backward-fn-signature + "
        "register-back-fn-after-wrap trifecta by wiring torch.log end-to-end through "
        "a tiny autograd."
    ),
}
emit_composite(spec_1)


# ================================================================= cx2
CX2_ATOMS = ["multiply-back", "unbroadcast-pattern", "arg-position-back-functions"]
spec_2 = {
    "atom_ids": CX2_ATOMS,
    "subtopics": _subs(CX2_ATOMS),
    "primary_atom": "multiply-back",
    "part": "part4",
    "exercise_index": 2,
    "exercise_title": "multiply_back0/back1 — per-arg chain rule + unbroadcast + argnum dispatch",
    "slug": "multiply-back-binary-with-unbroadcast-and-argnum",
    "atom_recap_md": (
        "## Binary-op back fns — three atoms in one expression\n"
        "\n"
        "Every binary op `out = f(x, y)` needs TWO back fns — one per arg-position —\n"
        "and each one ends with an unbroadcast step. Composition:\n"
        "\n"
        "1. **`arg-position-back-functions`** — argnum=0 returns `dL/dx`, argnum=1\n"
        "   returns `dL/dy`. For `multiply`, the two bodies differ only in which parent\n"
        "   appears: `grad_out * y` vs `grad_out * x`.\n"
        "2. **`multiply-back`** — the per-arg chain rule: `d(x*y)/dx = y`, so the raw\n"
        "   chain step is `grad_out * y` (and symmetrically for arg-1).\n"
        "3. **`unbroadcast-pattern`** — wrap the chain-rule result in `unbroadcast(grad,\n"
        "   parent)` so it matches the *pre-broadcast* parent shape. Without this, the\n"
        "   grad would have the broadcasted (output) shape and node `x` couldn't\n"
        "   accumulate it.\n"
        "\n"
        "The whole composite collapses to one expression per arg:\n"
        "`unbroadcast(grad_out * other_parent, this_parent)`.\n"
    ),
    "prompt_body": (
        "Build a single `cx2_multiply_back(grad_out, out, x, y, argnum)` that dispatches\n"
        "on `argnum` (0 or 1) and returns the gradient w.r.t. the requested parent,\n"
        "**already unbroadcast** to the parent's original shape.\n"
        "\n"
        "Use the provided `unbroadcast(grad, original)` helper (in the stub).\n"
        "\n"
        "Required behaviour:\n"
        "- `argnum=0` → return `dL/dx`, shape == `x.shape`.\n"
        "- `argnum=1` → return `dL/dy`, shape == `y.shape`.\n"
        "- Any other `argnum` → `raise ValueError`.\n"
        "- Broadcasting case: when `x.shape != y.shape != out.shape`, the returned grad\n"
        "  must be summed along the axes that were broadcast.\n"
    ),
    "stub_body": (
        "def unbroadcast(grad, original):\n"
        "    # peel leading axes, then collapse expanded size-1 axes\n"
        "    while grad.ndim > original.ndim:\n"
        "        grad = grad.sum(dim=0)\n"
        "    for i, size in enumerate(original.shape):\n"
        "        if size == 1 and grad.shape[i] != 1:\n"
        "            grad = grad.sum(dim=i, keepdim=True)\n"
        "    return grad\n"
        "\n"
        "\n"
        "def cx2_multiply_back(grad_out, out, x, y, argnum):\n"
        "    \"\"\"Dispatch on argnum; return chain-ruled + unbroadcast grad.\"\"\"\n"
        "    raise NotImplementedError()\n"
    ),
    "test_body": (
        "# --- (a) argnum dispatch: same-shape vector ---\n"
        "x = t.tensor([2.0, 3.0, 4.0])\n"
        "y = t.tensor([5.0, 6.0, 7.0])\n"
        "out = x * y\n"
        "g0 = cx2_multiply_back(t.ones(3), out, x, y, argnum=0)\n"
        "g1 = cx2_multiply_back(t.ones(3), out, x, y, argnum=1)\n"
        "assert g0.shape == x.shape and g1.shape == y.shape\n"
        "assert t.allclose(g0, y), f'argnum=0 chain-rule: got {g0}, expected y={y}'\n"
        "assert t.allclose(g1, x), f'argnum=1 chain-rule: got {g1}, expected x={x}'\n"
        "\n"
        "# --- (b) chain rule with non-unit grad_out ---\n"
        "grad_out = t.tensor([10.0, 100.0, 1000.0])\n"
        "g0 = cx2_multiply_back(grad_out, out, x, y, argnum=0)\n"
        "g1 = cx2_multiply_back(grad_out, out, x, y, argnum=1)\n"
        "assert t.allclose(g0, grad_out * y)\n"
        "assert t.allclose(g1, grad_out * x)\n"
        "\n"
        "# --- (c) broadcasting: x is (1,4), y is (3,4) → unbroadcast collapses axis 0 of x ---\n"
        "x_b = t.tensor([[1.0, 2.0, 3.0, 4.0]])\n"
        "y_b = t.tensor([[5.0, 6.0, 7.0, 8.0],\n"
        "                [9.0, 10.0, 11.0, 12.0],\n"
        "                [13.0, 14.0, 15.0, 16.0]])\n"
        "out_b = x_b * y_b\n"
        "g0_b = cx2_multiply_back(t.ones(3, 4), out_b, x_b, y_b, argnum=0)\n"
        "g1_b = cx2_multiply_back(t.ones(3, 4), out_b, x_b, y_b, argnum=1)\n"
        "assert g0_b.shape == x_b.shape, f'unbroadcast missing? g0 shape {g0_b.shape} vs x_b.shape {x_b.shape}'\n"
        "assert g1_b.shape == y_b.shape\n"
        "# x_b gradient = sum of y_b along the broadcast axis (axis 0), keepdim=True\n"
        "expected_g0_b = y_b.sum(dim=0, keepdim=True)\n"
        "assert t.allclose(g0_b, expected_g0_b), f'unbroadcast value wrong: {g0_b} vs {expected_g0_b}'\n"
        "assert t.allclose(g1_b, x_b.expand_as(y_b))  # x_b broadcast = each row equals x_b\n"
        "\n"
        "# --- (d) extra leading axis: x is (4,), y is (3,4) ---\n"
        "x_l = t.tensor([1.0, 2.0, 3.0, 4.0])\n"
        "y_l = t.tensor([[5.0, 6.0, 7.0, 8.0],\n"
        "                [9.0, 10.0, 11.0, 12.0],\n"
        "                [13.0, 14.0, 15.0, 16.0]])\n"
        "out_l = x_l * y_l\n"
        "g0_l = cx2_multiply_back(t.ones(3, 4), out_l, x_l, y_l, argnum=0)\n"
        "assert g0_l.shape == x_l.shape, f'leading-axis unbroadcast: {g0_l.shape} vs {x_l.shape}'\n"
        "assert t.allclose(g0_l, y_l.sum(dim=0))\n"
        "\n"
        "# --- (e) argnum dispatch: invalid argnum must raise ---\n"
        "try:\n"
        "    cx2_multiply_back(t.ones(3), out, x, y, argnum=2)\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('argnum=2 must raise ValueError')\n"
    ),
    "solution_body": (
        "def unbroadcast(grad, original):\n"
        "    while grad.ndim > original.ndim:\n"
        "        grad = grad.sum(dim=0)\n"
        "    for i, size in enumerate(original.shape):\n"
        "        if size == 1 and grad.shape[i] != 1:\n"
        "            grad = grad.sum(dim=i, keepdim=True)\n"
        "    return grad\n"
        "\n"
        "def cx2_multiply_back(grad_out, out, x, y, argnum):\n"
        "    if argnum == 0:\n"
        "        return unbroadcast(grad_out * y, x)   # dL/dx = grad_out * y, then unbroadcast\n"
        "    if argnum == 1:\n"
        "        return unbroadcast(grad_out * x, y)   # dL/dy = grad_out * x, then unbroadcast\n"
        "    raise ValueError(f'argnum must be 0 or 1, got {argnum}')\n"
    ),
    "solution_notes": (
        "All three atoms live in the same expression: the `argnum` branch is the\n"
        "arg-position dispatch; `grad_out * y` (or `* x`) is the multiply-back chain\n"
        "rule; `unbroadcast(..., parent)` is the unbroadcast pattern. Dropping any one\n"
        "of them breaks at least one of the (b), (c)/(d), (e) test cases."
    ),
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["multiply-back", "unbroadcast-pattern", "arg-position-back-functions"],
    "lo": (
        "Apply the binary-op back-fn pattern by composing multiply_back's per-arg "
        "chain rule with unbroadcast and argnum-based dispatch into a single function."
    ),
}
emit_composite(spec_2)


# ================================================================= cx3
CX3_ATOMS = ["multiply-back", "arg-position-back-functions"]
spec_3 = {
    "atom_ids": CX3_ATOMS,
    "subtopics": _subs(CX3_ATOMS),
    "primary_atom": "multiply-back",
    "part": "part4",
    "exercise_index": 3,
    "exercise_title": "mul_back0 vs mul_back1 — argnum dispatch (no broadcasting)",
    "slug": "multiply-back-argnum-dispatch",
    "atom_recap_md": (
        "## `multiply_back0` vs `multiply_back1` — argnum dispatch\n"
        "\n"
        "When the dispatcher hits `out = x * y` on the reverse pass, it has to choose\n"
        "WHICH back-fn to call based on which parent it's propagating into:\n"
        "\n"
        "- For parent `x` (argnum=0): chain rule gives `dL/dx = grad_out * y`.\n"
        "- For parent `y` (argnum=1): chain rule gives `dL/dy = grad_out * x`.\n"
        "\n"
        "The two bodies are NOT the same function — they use different parents. This\n"
        "atom-pair isolates the **dispatch on argnum** without the unbroadcast step\n"
        "(parents share the same shape here). Compare to `add`, where `back0` and\n"
        "`back1` happen to have the same body — but they STILL get registered\n"
        "separately at argnum=0 and argnum=1, because the dispatcher knows nothing\n"
        "about which ops happen to be symmetric.\n"
    ),
    "prompt_body": (
        "Assume `x.shape == y.shape == out.shape` — no broadcasting in this drill.\n"
        "\n"
        "Implement `cx3_mul_back(grad_out, out, x, y, argnum)` that returns:\n"
        "- `grad_out * y` when `argnum == 0`,\n"
        "- `grad_out * x` when `argnum == 1`,\n"
        "- raises `ValueError` for any other argnum.\n"
        "\n"
        "The argnum dispatch must select the correct *parent* to multiply against —\n"
        "the test asserts the two branches are not the same function by feeding it\n"
        "asymmetric `x` and `y`.\n"
    ),
    "stub_body": (
        "def cx3_mul_back(grad_out, out, x, y, argnum):\n"
        "    \"\"\"argnum=0 → dL/dx; argnum=1 → dL/dy; else ValueError.\"\"\"\n"
        "    raise NotImplementedError()\n"
    ),
    "test_body": (
        "# --- argnum=0 picks y, argnum=1 picks x (and they differ) ---\n"
        "x = t.tensor([2.0, 3.0, 4.0])\n"
        "y = t.tensor([10.0, 100.0, 1000.0])\n"
        "out = x * y\n"
        "g0 = cx3_mul_back(t.ones(3), out, x, y, argnum=0)\n"
        "g1 = cx3_mul_back(t.ones(3), out, x, y, argnum=1)\n"
        "assert t.allclose(g0, y), f'argnum=0 must equal y for unit grad_out: {g0}'\n"
        "assert t.allclose(g1, x), f'argnum=1 must equal x for unit grad_out: {g1}'\n"
        "assert not t.allclose(g0, g1), 'argnum=0 and argnum=1 must dispatch to different bodies'\n"
        "\n"
        "# --- chain rule with non-unit grad_out ---\n"
        "grad_out = t.tensor([5.0, -3.0, 2.0])\n"
        "assert t.allclose(cx3_mul_back(grad_out, out, x, y, argnum=0), grad_out * y)\n"
        "assert t.allclose(cx3_mul_back(grad_out, out, x, y, argnum=1), grad_out * x)\n"
        "\n"
        "# --- matrix shape preserved (no broadcasting) ---\n"
        "rng = t.Generator().manual_seed(0)\n"
        "X = t.randn(3, 4, generator=rng)\n"
        "Y = t.randn(3, 4, generator=rng)\n"
        "OUT = X * Y\n"
        "G = t.randn(3, 4, generator=rng)\n"
        "assert t.allclose(cx3_mul_back(G, OUT, X, Y, argnum=0), G * Y)\n"
        "assert t.allclose(cx3_mul_back(G, OUT, X, Y, argnum=1), G * X)\n"
        "\n"
        "# --- invalid argnum ---\n"
        "try:\n"
        "    cx3_mul_back(t.ones(3), out, x, y, argnum=99)\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('argnum=99 must raise ValueError')\n"
        "\n"
        "# --- witness vs torch.autograd ---\n"
        "x_ref = t.tensor([1.5, -0.3, 2.7], requires_grad=True)\n"
        "y_ref = t.tensor([0.8, 4.0, -1.2], requires_grad=True)\n"
        "(x_ref * y_ref).sum().backward()\n"
        "ours0 = cx3_mul_back(t.ones(3), x_ref.detach() * y_ref.detach(), x_ref.detach(), y_ref.detach(), 0)\n"
        "ours1 = cx3_mul_back(t.ones(3), x_ref.detach() * y_ref.detach(), x_ref.detach(), y_ref.detach(), 1)\n"
        "assert t.allclose(ours0, x_ref.grad, atol=1e-6)\n"
        "assert t.allclose(ours1, y_ref.grad, atol=1e-6)\n"
    ),
    "solution_body": (
        "def cx3_mul_back(grad_out, out, x, y, argnum):\n"
        "    if argnum == 0:\n"
        "        return grad_out * y            # d(x*y)/dx = y\n"
        "    if argnum == 1:\n"
        "        return grad_out * x            # d(x*y)/dy = x\n"
        "    raise ValueError(f'argnum must be 0 or 1, got {argnum}')\n"
    ),
    "solution_notes": (
        "The `argnum` branches encode `arg-position-back-functions`; the actual chain\n"
        "rule bodies (`grad_out * y` and `grad_out * x`) encode `multiply-back`. Drop\n"
        "the branching and you can't dispatch; drop the chain rule and you'd return\n"
        "`grad_out` unchanged (which is `add_back`, not `multiply_back`)."
    ),
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["multiply-back", "arg-position-back-functions"],
    "lo": (
        "Apply argnum-based dispatch on multiply_back to select the correct per-arg "
        "chain-rule body without broadcasting concerns."
    ),
}
emit_composite(spec_3)


# ================================================================= cx4
CX4_ATOMS = ["exp-back", "backward-fn-signature"]
spec_4 = {
    "atom_ids": CX4_ATOMS,
    "subtopics": _subs(CX4_ATOMS),
    "primary_atom": "exp-back",
    "part": "part4",
    "exercise_index": 4,
    "exercise_title": "exp_back — use cached out instead of recomputing exp(x)",
    "slug": "exp-back-via-cached-out",
    "atom_recap_md": (
        "## `exp_back` via cached `out` — two atoms in one one-liner\n"
        "\n"
        "1. **`backward-fn-signature`** — every back fn takes `(grad_out, out, *args)`.\n"
        "   The `out` slot is the cached forward result; it's there *specifically* so\n"
        "   that back fns can reuse it instead of recomputing the activation.\n"
        "2. **`exp-back`** — the local derivative of `exp(x)` is `exp(x)`, which IS\n"
        "   `out`. So the chain rule collapses to `grad_in = grad_out * out`. No\n"
        "   second exp call needed.\n"
        "\n"
        "Composition: the signature *gives* you `out`, and the math *requires* you to\n"
        "use it. If you instead wrote `grad_out * t.exp(x)` you'd get the same number\n"
        "(modulo float rounding) but pay for the activation twice. The test below\n"
        "passes a deliberately WRONG `out` to catch implementations that secretly\n"
        "recompute `exp(x)`.\n"
    ),
    "prompt_body": (
        "Implement `cx4_exp_back(grad_out, out, x)` for `out = exp(x)`.\n"
        "\n"
        "Requirements:\n"
        "- Follow the canonical `(grad_out, out, x) -> grad_in` signature.\n"
        "- Return `grad_out * out` — **use the cached `out`**, do NOT call `torch.exp`,\n"
        "  `np.exp`, `math.exp`, or `**` with base e inside the function. The test\n"
        "  passes a fake `out` and checks that the result tracks the fake (i.e. you\n"
        "  trust `out`, not `x`).\n"
        "- Return shape must equal `x.shape` (which equals `out.shape` for exp).\n"
    ),
    "stub_body": (
        "def cx4_exp_back(grad_out, out, x):\n"
        "    \"\"\"dL/dx for out = exp(x). MUST use cached out, not recompute exp(x).\"\"\"\n"
        "    raise NotImplementedError()\n"
    ),
    "test_body": (
        "# --- real-world case: out = exp(x), grad_out = ones → grad_in = out ---\n"
        "x = t.tensor([0.0, 1.0, 2.0])\n"
        "out = t.exp(x)\n"
        "g = cx4_exp_back(t.ones(3), out, x)\n"
        "assert g.shape == x.shape\n"
        "assert t.allclose(g, out), f'unit grad_out: {g} vs {out}'\n"
        "\n"
        "# --- non-unit grad_out: chain rule scales each entry ---\n"
        "grad_out = t.tensor([3.0, -2.0, 5.0])\n"
        "g = cx4_exp_back(grad_out, out, x)\n"
        "assert t.allclose(g, grad_out * out)\n"
        "\n"
        "# --- matrix shape ---\n"
        "rng = t.Generator().manual_seed(1)\n"
        "X = t.randn(3, 4, generator=rng)\n"
        "G = t.randn(3, 4, generator=rng)\n"
        "OUT = t.exp(X)\n"
        "g = cx4_exp_back(G, OUT, X)\n"
        "assert g.shape == (3, 4)\n"
        "assert t.allclose(g, G * OUT)\n"
        "\n"
        "# --- THE point: must use cached `out`, not recompute exp(x) ---\n"
        "# Pass a DELIBERATELY WRONG `out` (not equal to exp(x)) and verify the result\n"
        "# tracks the fake `out`. A recompute-from-x implementation would ignore it.\n"
        "fake_x = t.tensor([0.0, 0.0, 0.0])      # exp(0) is 1.0, but we lie:\n"
        "fake_out = t.tensor([0.25, 0.5, 0.75])\n"
        "got = cx4_exp_back(t.ones(3), fake_out, fake_x)\n"
        "assert t.allclose(got, fake_out), (\n"
        "    f'implementation appears to recompute exp(x) rather than trust out: got {got}, '\n"
        "    f'expected {fake_out} (chain rule on the fake out)'\n"
        ")\n"
        "\n"
        "# --- witness vs torch.autograd ---\n"
        "x_ref = t.tensor([-0.5, 0.0, 0.8, 1.5], requires_grad=True)\n"
        "t.exp(x_ref).sum().backward()\n"
        "out_cached = t.exp(x_ref.detach())\n"
        "ours = cx4_exp_back(t.ones(4), out_cached, x_ref.detach())\n"
        "assert t.allclose(ours, x_ref.grad, atol=1e-6)\n"
    ),
    "solution_body": (
        "def cx4_exp_back(grad_out, out, x):\n"
        "    # d/dx exp(x) = exp(x) = out (already cached by the forward pass).\n"
        "    # Signature passes `out` precisely so we can reuse it — no recompute.\n"
        "    return grad_out * out\n"
    ),
    "solution_notes": (
        "Two atoms in one line: the parameter list IS the canonical back-fn signature\n"
        "(`backward-fn-signature`), and `grad_out * out` is the cached-out form of\n"
        "exp's chain rule (`exp-back`). The fake-out test fails any implementation\n"
        "that quietly recomputes `t.exp(x)` instead of trusting `out`."
    ),
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["exp-back", "backward-fn-signature", "back-fn-uses-cached-out"],
    "lo": (
        "Apply the canonical backward-fn signature and the cached-out optimisation "
        "together by writing exp_back as grad_out * out (no second exp call)."
    ),
}
emit_composite(spec_4)


# ================================================================= cx5
CX5_ATOMS = ["log-back", "chain-rule-elementwise"]
spec_5 = {
    "atom_ids": CX5_ATOMS,
    "subtopics": _subs(CX5_ATOMS),
    "primary_atom": "log-back",
    "part": "part4",
    "exercise_index": 5,
    "exercise_title": "log_back — one-line elementwise chain rule grad_out / x",
    "slug": "log-back-elementwise-chain-rule",
    "atom_recap_md": (
        "## `log_back` as the canonical elementwise chain rule\n"
        "\n"
        "1. **`chain-rule-elementwise`** — for any elementwise op `out = f(x)`, the\n"
        "   Jacobian is diagonal, so `dL/dx[i] = dL/dout[i] * f'(x[i])` — no matrix\n"
        "   materialised, just a per-position product.\n"
        "2. **`log-back`** — instantiate that rule for `f = log`, where `f'(x) = 1/x`.\n"
        "   The whole back fn collapses to `grad_out / x`.\n"
        "\n"
        "Composition: this is the shortest possible non-trivial back fn — the chain\n"
        "rule (atom 1) is what TELLS you to multiply by the local derivative; the\n"
        "specific `1/x` (atom 2) is what TELLS you the derivative of log.\n"
    ),
    "prompt_body": (
        "Implement `cx5_log_back(grad_out, out, x)` for `out = log(x)`.\n"
        "\n"
        "One line. Use the elementwise chain rule: `grad_in = grad_out * f'(x)` with\n"
        "`f'(x) = 1/x`, which collapses to `grad_out / x`.\n"
        "\n"
        "Constraints:\n"
        "- Do NOT materialize the diagonal Jacobian (no `torch.diag`, no `eye`).\n"
        "- Return shape must equal `x.shape`.\n"
        "- `out` is part of the signature but unused (log doesn't need cached out).\n"
    ),
    "stub_body": (
        "def cx5_log_back(grad_out, out, x):\n"
        "    \"\"\"dL/dx for out = log(x), via the elementwise chain rule.\"\"\"\n"
        "    raise NotImplementedError()\n"
    ),
    "test_body": (
        "# --- scalar ---\n"
        "x = t.tensor([2.0])\n"
        "out = t.log(x)\n"
        "g = cx5_log_back(t.tensor([1.0]), out, x)\n"
        "assert t.allclose(g, t.tensor([0.5])), f'scalar: {g}'\n"
        "\n"
        "# --- vector + non-unit grad_out (true chain-rule scaling) ---\n"
        "x = t.tensor([1.0, 2.0, 4.0])\n"
        "out = t.log(x)\n"
        "grad_out = t.tensor([5.0, -3.0, 2.0])\n"
        "g = cx5_log_back(grad_out, out, x)\n"
        "expected = grad_out / x\n"
        "assert g.shape == x.shape\n"
        "assert t.allclose(g, expected), f'vector chain-rule: {g} vs {expected}'\n"
        "\n"
        "# --- matrix ---\n"
        "rng = t.Generator().manual_seed(0)\n"
        "X = t.rand(3, 4, generator=rng) + 0.5\n"
        "G = t.randn(3, 4, generator=rng)\n"
        "OUT = t.log(X)\n"
        "g = cx5_log_back(G, OUT, X)\n"
        "assert g.shape == (3, 4)\n"
        "assert t.allclose(g, G / X)\n"
        "\n"
        "# --- shape must follow x.shape, not out.shape (they coincide here, but assert) ---\n"
        "assert g.shape == X.shape\n"
        "\n"
        "# --- witness vs torch.autograd ---\n"
        "x_ref = t.tensor([0.5, 1.0, 2.5, 7.0], requires_grad=True)\n"
        "t.log(x_ref).sum().backward()\n"
        "ours = cx5_log_back(t.ones(4), t.log(x_ref.detach()), x_ref.detach())\n"
        "assert t.allclose(ours, x_ref.grad, atol=1e-6)\n"
        "\n"
        "# --- chain-rule semantics: zero grad_out → zero grad_in (sanity) ---\n"
        "z = cx5_log_back(t.zeros(3), t.log(t.tensor([1.0, 2.0, 4.0])), t.tensor([1.0, 2.0, 4.0]))\n"
        "assert t.allclose(z, t.zeros(3)), 'zero grad_out must propagate to zero grad_in'\n"
    ),
    "solution_body": (
        "def cx5_log_back(grad_out, out, x):\n"
        "    # Elementwise chain rule: grad_in = grad_out * f'(x), with f'(x) = 1/x.\n"
        "    return grad_out / x\n"
    ),
    "solution_notes": (
        "The `/ x` is `log-back` (specific derivative `1/x`); the per-position\n"
        "broadcasting multiplication with `grad_out` is the elementwise chain rule.\n"
        "Note `out` is unused — that's fine, the signature is uniform across all back\n"
        "fns even when individual ones don't need every field."
    ),
    "bloom_level": "Apply",
    "difficulty_num": 1,
    "kcs": ["log-back", "chain-rule-elementwise"],
    "lo": (
        "Apply the elementwise chain rule with f'(x) = 1/x to produce log_back as the "
        "one-line expression grad_out / x."
    ),
}
emit_composite(spec_5)


# ================================================================= cx6
CX6_ATOMS = ["negative-back", "backward-fn-signature"]
spec_6 = {
    "atom_ids": CX6_ATOMS,
    "subtopics": _subs(CX6_ATOMS),
    "primary_atom": "negative-back",
    "part": "part4",
    "exercise_index": 6,
    "exercise_title": "negative_back — −grad_out under the canonical (grad_out, out, x) contract",
    "slug": "negative-back-signature-contract",
    "atom_recap_md": (
        "## `negative_back` as a minimal signature-contract demo\n"
        "\n"
        "1. **`backward-fn-signature`** — every back fn takes `(grad_out, out, *args)`\n"
        "   and returns `dL/dargs[i]` with the SAME shape as the corresponding arg.\n"
        "   The signature is uniform even when some fields go unused.\n"
        "2. **`negative-back`** — `out = -x` has constant local derivative `-1`, so\n"
        "   the chain rule collapses to `dL/dx = -grad_out`. Neither `out` nor `x` is\n"
        "   read.\n"
        "\n"
        "Composition: `negative_back` is the SIMPLEST possible back fn — useful\n"
        "exactly because it forces you to honour the full signature even when half of\n"
        "it is dead weight. The reverse-pass dispatcher must be able to call any back\n"
        "fn the same way, so the signature contract is non-negotiable.\n"
    ),
    "prompt_body": (
        "Implement `cx6_negative_back(grad_out, out, x)` for `out = -x`.\n"
        "\n"
        "Requirements:\n"
        "- Signature must be exactly `(grad_out, out, x)` — three positional args.\n"
        "  The test calls it positionally AND introspects its signature.\n"
        "- Return `-grad_out` (chain rule with constant local derivative `-1`).\n"
        "- Must NOT mutate `grad_out` in place (the test re-uses the same tensor).\n"
        "- Output shape must equal `x.shape` (which here equals `grad_out.shape`).\n"
        "- Do not read `out` or `x` for the value — the local derivative is\n"
        "  position-independent.\n"
    ),
    "stub_body": (
        "def cx6_negative_back(grad_out, out, x):\n"
        "    \"\"\"dL/dx for out = -x. Honours (grad_out, out, x) signature.\"\"\"\n"
        "    raise NotImplementedError()\n"
    ),
    "test_body": (
        "import inspect\n"
        "# --- signature contract: three positional params named (grad_out, out, x) ---\n"
        "sig = inspect.signature(cx6_negative_back)\n"
        "params = list(sig.parameters)\n"
        "assert params == ['grad_out', 'out', 'x'], (\n"
        "    f'signature must be (grad_out, out, x), got {params}'\n"
        ")\n"
        "\n"
        "# --- scalar ---\n"
        "x = t.tensor([3.0])\n"
        "out = -x\n"
        "g = cx6_negative_back(t.tensor([1.0]), out, x)\n"
        "assert t.allclose(g, t.tensor([-1.0])), f'scalar: {g}'\n"
        "\n"
        "# --- vector + non-unit grad_out ---\n"
        "x = t.tensor([1.0, -2.0, 3.0])\n"
        "out = -x\n"
        "grad_out = t.tensor([5.0, 7.0, -2.0])\n"
        "g = cx6_negative_back(grad_out, out, x)\n"
        "assert g.shape == x.shape\n"
        "assert t.allclose(g, -grad_out), f'vector: {g}'\n"
        "\n"
        "# --- matrix shape ---\n"
        "rng = t.Generator().manual_seed(0)\n"
        "X = t.randn(3, 4, generator=rng)\n"
        "G = t.randn(3, 4, generator=rng)\n"
        "g = cx6_negative_back(G, -X, X)\n"
        "assert g.shape == (3, 4)\n"
        "assert t.allclose(g, -G)\n"
        "\n"
        "# --- non-mutation: grad_out must be unchanged by the call ---\n"
        "grad_in = t.tensor([1.0, 2.0, 3.0])\n"
        "grad_in_copy = grad_in.clone()\n"
        "_ = cx6_negative_back(grad_in, t.zeros(3), t.zeros(3))\n"
        "assert t.allclose(grad_in, grad_in_copy), 'must not mutate grad_out in place'\n"
        "\n"
        "# --- doesn't depend on out or x: scrambling them must not change the result ---\n"
        "go = t.tensor([4.0, -1.0, 8.0])\n"
        "real_x = t.tensor([1.0, 2.0, 3.0])\n"
        "g_real = cx6_negative_back(go, -real_x, real_x)\n"
        "g_fake = cx6_negative_back(go, t.tensor([99.0, 99.0, 99.0]), t.tensor([-7.0, 0.5, 1e6]))\n"
        "assert t.allclose(g_real, g_fake), (\n"
        "    'negative_back must depend ONLY on grad_out — local derivative is constant -1'\n"
        ")\n"
        "\n"
        "# --- witness vs torch.autograd ---\n"
        "x_ref = t.tensor([1.5, -0.3, 2.7], requires_grad=True)\n"
        "(-x_ref).sum().backward()\n"
        "ours = cx6_negative_back(t.ones(3), -x_ref.detach(), x_ref.detach())\n"
        "assert t.allclose(ours, x_ref.grad, atol=1e-6)\n"
    ),
    "solution_body": (
        "def cx6_negative_back(grad_out, out, x):\n"
        "    # d/dx (-x) = -1; chain rule → -grad_out. `out` and `x` go unread.\n"
        "    return -grad_out\n"
    ),
    "solution_notes": (
        "Two atoms: the parameter list IS the signature contract\n"
        "(`backward-fn-signature`); the body `-grad_out` IS `negative-back`. The\n"
        "scramble test confirms the body ignores `out`/`x`, while the introspection\n"
        "test confirms the signature is still the canonical three-positional form."
    ),
    "bloom_level": "Apply",
    "difficulty_num": 1,
    "kcs": ["negative-back", "backward-fn-signature"],
    "lo": (
        "Apply the canonical (grad_out, out, x) backward-fn contract to the simplest "
        "case — negative_back — and verify the signature is honoured even when the "
        "body ignores most of it."
    ),
}
emit_composite(spec_6)


if __name__ == "__main__":
    print("authored cx1..cx6 composites under arena-procedural-drills/composites/part4/")
