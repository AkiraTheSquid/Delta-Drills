#!/usr/bin/env python3
"""Author 8 deepening drills for high-ARENA-usage atoms.

Each new exercise (ex3 or ex2) probes a DISTINCT facet from the existing
drill(s) for that atom — different cognitive operation, different surface
context, different difficulty rung. ONE LO + ONE Bloom + <=2 KCs per drill.

Verification re-runs each spec's solution against its test_body inside the
build venv (torch 2.12.0+cpu) before any notebook is emitted.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

# ---------------------------------------------------------------------------
# Per-atom recap blocks (reused from prior batches but trimmed to deepening focus).
# ---------------------------------------------------------------------------

RECAP_TRAINING_STEP_CYCLE = (
    "## Training-step cycle — quick refresher\n"
    "\n"
    "```\n"
    "logits = model(x)             # 1. forward\n"
    "loss   = loss_fn(logits, y)   # 2. loss\n"
    "loss.backward()               # 3. backward → grads into .grad\n"
    "optimizer.step()              # 4. apply update from .grad\n"
    "optimizer.zero_grad()         # 5. clear .grad for next batch\n"
    "```\n"
    "\n"
    "**Two ordering invariants.** `backward` must come *before* `step` — "
    "otherwise `.grad` is `None` (or stale from a previous batch) and `step` "
    "either crashes or applies the wrong update. `zero_grad` must come *after* "
    "`step` (or before next forward) so gradients don't accumulate across "
    "batches. Swapping `backward` and `step` is the silent-failure variant: "
    "no error is raised, but training silently uses the previous step's grad."
)

RECAP_CONTIGUOUS_LAYOUT = (
    "## Contiguous layout — quick refresher\n"
    "\n"
    "A tensor is **contiguous** iff its stride tuple matches the row-major "
    "formula derived from its shape:\n"
    "```\n"
    "stride[-1] = 1\n"
    "stride[k]  = stride[k+1] * shape[k+1]   # walk right-to-left\n"
    "```\n"
    "`x.is_contiguous()` checks exactly that. `view` requires it; "
    "`reshape` falls back to a copy when it isn't satisfied.\n"
    "\n"
    "Operations that BREAK contiguity without copying memory:\n"
    "- `transpose` / `permute` — swap strides only\n"
    "- `t[::2]` / `expand` — set a stride to 0 or to a multiple of element size\n"
    "Operations that RESTORE contiguity by copying:\n"
    "- `.contiguous()` — explicit\n"
    "- `.reshape(...)` — implicit when needed"
)

RECAP_AS_STRIDED = (
    "## `as_strided` — quick refresher\n"
    "\n"
    "`t.as_strided(x, size, stride)` re-interprets `x`'s underlying storage "
    "with a new shape + stride tuple. **It never allocates.** The arguments "
    "are in *elements*, not bytes. The view aliases the same storage — writes "
    "through one alias are visible through the other.\n"
    "\n"
    "**Window-stride vs in-window-stride.** A 1-D sliding window of width `K` "
    "stepping by 1 element has shape `(L_out, K)` and stride `(s, s)` where "
    "`s = x.stride(0)`. To dilate or sub-sample the windows, you multiply ONE "
    "of those by an integer factor:\n"
    "- `stride = (s, s*d)`     → dilated window (gaps INSIDE the window)\n"
    "- `stride = (s*step, s)`  → strided windows (gaps BETWEEN windows)\n"
    "Compute `L_out` from the same formula PyTorch's conv uses: "
    "`L_out = (L - dilated_K) // step + 1`."
)

RECAP_TENSOR_TO_DEVICE = (
    "## `tensor.to(device)` — quick refresher\n"
    "\n"
    "`x.to(device)` is **not in-place** for tensors — it returns a (possibly "
    "new) tensor on the target device. You must capture the return value. "
    "Contrast with `model.to(device)`, which IS in-place on the Module's "
    "parameters/buffers.\n"
    "\n"
    "**Device-mismatch errors.** Any binary op (add, matmul, ...) between a "
    "CPU tensor and a CUDA tensor raises `RuntimeError: Expected all tensors "
    "to be on the same device`. The fix is to move every input to one common "
    "device BEFORE the op. The idiomatic helper aligns a *list* of tensors "
    "in one pass.\n"
    "\n"
    "**Idempotence.** `x.to(x.device)` returns `x` itself (same object). "
    "`x.to(other_device)` returns a fresh tensor. The `is` check exposes this."
)

RECAP_STACK_VS_CAT = (
    "## `t.stack` vs `t.cat` — quick refresher\n"
    "\n"
    "- `t.stack(seq, dim=k)`: every tensor in `seq` has the same shape `S`; "
    "result has shape `S` with a new axis of size `len(seq)` INSERTED at "
    "position `k`. Rank goes UP by 1.\n"
    "- `t.cat(seq, dim=k)`: every tensor has the same shape EXCEPT possibly "
    "axis `k`; result extends axis `k`. Rank STAYS the same.\n"
    "\n"
    "**Round-trip identity.** `t.stack(seq, dim=k)` is equivalent to "
    "`t.cat([s.unsqueeze(k) for s in seq], dim=k)`. Inverting it: "
    "`t.stack(...).unbind(dim=k)` returns the original tuple.\n"
    "\n"
    "**Interleaving.** Given two equal-length 1-D tensors `a, b`, the "
    "interleaved tensor `[a0, b0, a1, b1, ...]` is `t.stack([a, b], dim=1)"
    ".reshape(-1)`. The stack inserts a 'pair' axis, the reshape walks it in "
    "the right order."
)

RECAP_LINALG_SOLVE = (
    "## `t.linalg.solve` — quick refresher\n"
    "\n"
    "`t.linalg.solve(A, b)` solves `A @ x == b` for `x`. Shape contract:\n"
    "- `A: (..., n, n)` — leading dims are the batch.\n"
    "- `b: (..., n)`    → returns `x: (..., n)`         (one RHS per batch)\n"
    "- `b: (..., n, m)` → returns `x: (..., n, m)`      (m RHS columns per batch)\n"
    "\n"
    "**Multiple right-hand sides.** When `b` carries a trailing `m` axis, "
    "the LU factorization of each `A[..., :, :]` is reused for all `m` "
    "substitutions. This is how you compute a matrix inverse efficiently "
    "(`solve(A, I)`) — and it's exactly the shape ARENA's triangle-mesh "
    "rasterization needs for batched barycentric solves with multiple query "
    "points per triangle."
)

RECAP_CONV_OUTPUT_SHAPE = (
    "## Conv output shape — quick refresher\n"
    "\n"
    "The standard formula (dilation=1):\n"
    "```\n"
    "L_out = (L_in + 2*P - K) // S + 1\n"
    "```\n"
    "**Inverting for 'SAME' padding.** Sometimes you want `L_out == L_in` "
    "(or `ceil(L_in / S)`). Solve the formula for `P`:\n"
    "```\n"
    "L_in + 2*P - K = (L_out - 1) * S\n"
    "P = ((L_out - 1) * S - L_in + K) / 2\n"
    "```\n"
    "With `S=1` this collapses to `P = (K - 1) // 2` for odd `K` — the "
    "classic 'half-kernel' padding everyone memorises. For `S > 1` or even "
    "`K`, the inversion isn't always an integer; you round up (top/right) "
    "and floor (bottom/left) to split the asymmetry. ARENA's `Conv2d` does "
    "NOT do this for you — the learner must compute the padding themselves."
)

RECAP_CONV_PADDING = (
    "## Conv zero-padding — quick refresher\n"
    "\n"
    "Zero-pad a `(B, IC, H, W)` input by allocating a zero-buffer sized to "
    "the padded extent, then assigning the original into the interior slice:\n"
    "```\n"
    "out = x.new_zeros(B, IC, top + H + bottom, left + W + right)\n"
    "out[..., top:top + H, left:left + W] = x\n"
    "```\n"
    "**Asymmetric pads.** Each of the four spatial sides can take a DIFFERENT "
    "amount. The single-int `padding=k` of `nn.Conv2d` is the symmetric "
    "shorthand for `top=bottom=left=right=k`; the 4-tuple form lets you "
    "match arbitrary input extents (e.g. 'SAME' padding for an even-stride "
    "conv often needs asymmetric padding).\n"
    "\n"
    "**Use `new_zeros`** (not `t.zeros`) so dtype/device inherit from `x`."
)


TOPIC_TENSOR_MECH = "prereqs_tensor_mechanics"
TOPIC_TRAINING_LOOP = "prereqs_training_loop"
TOPIC_GEOMETRY_CNN = "prereqs_geometry_cnn"


SPECS = [
    # ===================================================================
    # training-step-cycle  —  ex3
    # NEW facet: silent-failure ordering bug (step BEFORE backward).
    # ex1 = order calls correctly. ex2 = forgot zero_grad (crash-ish).
    # This one = swapped order. No exception is raised — only the loss
    # trajectory reveals the bug. Bloom: Analyze.
    # ===================================================================
    {
        "atom_id": "training-step-cycle",
        "subtopic": "PyTorch: Training step cycle",
        "topic_folder": TOPIC_TRAINING_LOOP,
        "atom_recap_md": RECAP_TRAINING_STEP_CYCLE,
        "exercise_index": 3,
        "exercise_title": "diagnose a step-before-backward ordering bug",
        "slug": "diagnose-a-step-before-backward-ordering-bug",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["debug", "ordering-bug", "stale-grad", "silent-failure"],
        "kcs": [
            "training-step-five-call-order",
            "training-step-debug-via-loss-trajectory",
        ],
        "lo": (
            "Analyze a training loop whose `optimizer.step()` is called BEFORE "
            "`loss.backward()` and fix the ordering so the loss curve becomes "
            "monotonically decreasing on a 1-parameter regression."
        ),
        "prompt_body": (
            "Below is `train_swapped` — a training loop where someone wrote "
            "`optimizer.step()` BEFORE `loss.backward()`. No Python exception "
            "is raised: on the first iteration `w.grad` is `None`, so "
            "`optimizer.step()` is a no-op; on every later iteration `step` "
            "consumes the previous iteration's gradient — one step BEHIND.\n\n"
            "Implement `ex3_train_fixed(w_init, x, y, lr, n_steps)` — the "
            "corrected version. Use the canonical 5-call order:\n"
            "  `forward → loss → backward → step → zero_grad`\n\n"
            "Return `(w_final, fixed_losses, swapped_losses)`:\n"
            "- `w_final`: a detached 1-element tensor of the trained weight.\n"
            "- `fixed_losses`: list of `n_steps` floats from YOUR loop.\n"
            "- `swapped_losses`: list of `n_steps` floats from `train_swapped` "
            "on the SAME inputs (call it yourself).\n\n"
            "Snapshot every loss BEFORE `backward()` so the trajectory is "
            "the pre-step value at iteration `i`. The test verifies your "
            "loop converges and the swapped loop lags behind."
        ),
        "stub": (
            "def train_swapped(w_init: float, x: Tensor, y: Tensor,\n"
            "                  lr: float, n_steps: int) -> tuple:\n"
            "    # BUG: step() is called BEFORE backward(). No exception\n"
            "    # raised — silent one-step lag for the whole training run.\n"
            "    w = t.tensor([w_init], requires_grad=True)\n"
            "    optimizer = t.optim.SGD([w], lr=lr)\n"
            "    losses = []\n"
            "    for _ in range(n_steps):\n"
            "        pred = w * x\n"
            "        loss = ((pred - y) ** 2).mean()\n"
            "        losses.append(loss.item())\n"
            "        optimizer.step()       # <-- WRONG: before backward\n"
            "        loss.backward()\n"
            "        optimizer.zero_grad()\n"
            "    return w.detach().clone(), losses\n"
            "\n"
            "\n"
            "def ex3_train_fixed(w_init: float, x: Tensor, y: Tensor,\n"
            "                    lr: float, n_steps: int) -> tuple:\n"
            '    """Return (w_final, fixed_losses, swapped_losses)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.tensor([1.0, 2.0, 3.0, 4.0])\n"
            "y = t.tensor([2.0, 4.0, 6.0, 8.0])   # true w = 2.0\n"
            "w_final, fixed_losses, swapped_losses = ex3_train_fixed(\n"
            "    w_init=0.0, x=x, y=y, lr=0.05, n_steps=20)\n"
            "\n"
            "assert isinstance(fixed_losses, list) and isinstance(swapped_losses, list)\n"
            "assert len(fixed_losses) == 20\n"
            "assert len(swapped_losses) == 20\n"
            "\n"
            "# Fixed loop must converge cleanly.\n"
            "for i in range(1, len(fixed_losses)):\n"
            "    assert fixed_losses[i] <= fixed_losses[i - 1] + 1e-7, (\n"
            "        f'fixed loss not decreasing at step {i}: '\n"
            "        f'{fixed_losses[i-1]:.6f} -> {fixed_losses[i]:.6f}; '\n"
            "        f'check call order in ex3_train_fixed.'\n"
            "    )\n"
            "assert abs(w_final.item() - 2.0) < 0.05, (\n"
            "    f'fixed loop expected w ~ 2.0, got {w_final.item():.4f}'\n"
            ")\n"
            "\n"
            "# Swapped loop must lag — first loss equals fixed first loss\n"
            "# (initial weights identical) but final loss must be NOTICEABLY\n"
            "# worse than the fixed one (one step behind for 20 iterations).\n"
            "assert abs(swapped_losses[0] - fixed_losses[0]) < 1e-6, (\n"
            "    'first iteration losses must agree — same w_init'\n"
            ")\n"
            "assert swapped_losses[-1] > fixed_losses[-1] + 1e-4, (\n"
            "    f'swapped final loss ({swapped_losses[-1]:.6f}) should LAG '\n"
            "    f'fixed final loss ({fixed_losses[-1]:.6f}); '\n"
            "    f'if these are equal you may have accidentally fixed the bug '\n"
            "    f'inside train_swapped — leave it broken on purpose.'\n"
            ")"
        ),
        "solution_body": (
            "def ex3_train_fixed(w_init, x, y, lr, n_steps):\n"
            "    w = t.tensor([w_init], requires_grad=True)\n"
            "    optimizer = t.optim.SGD([w], lr=lr)\n"
            "    fixed_losses = []\n"
            "    for _ in range(n_steps):\n"
            "        pred = w * x\n"
            "        loss = ((pred - y) ** 2).mean()\n"
            "        fixed_losses.append(loss.item())\n"
            "        loss.backward()\n"
            "        optimizer.step()\n"
            "        optimizer.zero_grad()\n"
            "    _, swapped_losses = train_swapped(w_init, x, y, lr, n_steps)\n"
            "    return w.detach().clone(), fixed_losses, swapped_losses"
        ),
        "solution_notes": (
            "**Why the swap is a SILENT bug.** On iteration 0, `w.grad is "
            "None` — `optimizer.step()` short-circuits with no update. On "
            "iteration 1, `step()` consumes iteration-0's gradient (computed "
            "AFTER step on iteration 0). The whole run is one step behind.\n\n"
            "**No exception is raised** because PyTorch tolerates `grad=None` "
            "in `step()` (it just skips that param). Only the loss "
            "trajectory exposes the lag — which is why ARENA repeatedly "
            "emphasizes 'log your loss every iteration and look at the "
            "curve, not just the final value.'\n\n"
            "**The cure is mechanical.** Memorize the order `forward → loss "
            "→ backward → step → zero_grad` as one atomic block. Every legit "
            "PyTorch training loop in the wild has this skeleton."
        ),
        "extra_imports": [],
    },

    # ===================================================================
    # contiguous-layout  —  ex3
    # NEW facet: classify contiguity from EXPLICIT shape+stride tuples,
    # without ever touching torch — pure stride arithmetic. ex1 was
    # 'compute strides from a shape', ex2 was 'fix view-after-transpose'.
    # Here: given (shape, stride), return is_contiguous(). Bloom: Analyze.
    # ===================================================================
    {
        "atom_id": "contiguous-layout",
        "subtopic": "PyTorch: Contiguous layout",
        "topic_folder": TOPIC_TENSOR_MECH,
        "atom_recap_md": RECAP_CONTIGUOUS_LAYOUT,
        "exercise_index": 3,
        "exercise_title": "classify contiguity from shape and stride tuples",
        "slug": "classify-contiguity-from-shape-and-stride-tuples",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["contiguous", "stride-classifier", "row-major"],
        "kcs": [
            "contiguous-stride-formula",
            "is-contiguous-check",
        ],
        "lo": (
            "Analyze a `(shape, stride)` pair without instantiating any "
            "tensor and return True iff the strides match the row-major "
            "contiguous formula."
        ),
        "prompt_body": (
            "Implement `ex3_is_contiguous_from_meta(shape, stride)`. Given a "
            "shape tuple and a stride tuple (both in *elements*, not bytes), "
            "return `True` iff a tensor with that metadata would have "
            "`is_contiguous() == True`.\n\n"
            "**Rules** (handles edge cases the way PyTorch does):\n"
            "1. Length mismatch ⇒ `False`.\n"
            "2. The 0-d / 1-d / empty cases: an empty shape `()` is "
            "vacuously contiguous (`True`). Any axis of size 0 ⇒ `True` "
            "(empty tensors are always contiguous in PyTorch).\n"
            "3. Otherwise compute the expected row-major strides:\n"
            "   `expected[-1] = 1`, `expected[k] = expected[k+1] * shape[k+1]`.\n"
            "4. Compare element-wise. **But:** any axis of size 1 is "
            "irrelevant — its stride doesn't matter (PyTorch treats it as a "
            "free dimension). Skip the comparison for axes where "
            "`shape[k] == 1`.\n\n"
            "Inputs: `shape: tuple[int, ...]`, `stride: tuple[int, ...]`.\n"
            "Output: `bool`.\n\n"
            "Do NOT build any tensor. This is a pure-arithmetic predicate."
        ),
        "stub": (
            "def ex3_is_contiguous_from_meta(shape: tuple, stride: tuple) -> bool:\n"
            '    """True iff (shape, stride) describes a row-major contiguous tensor."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Pull live ground truth out of torch — every (shape, stride) we test\n"
            "# must agree with what an actual tensor reports.\n"
            "def _gt(shape, stride):\n"
            "    # Build a synthetic tensor with this exact metadata via as_strided.\n"
            "    if any(s == 0 for s in shape):\n"
            "        return True   # PyTorch treats empty tensors as contiguous\n"
            "    storage_n = 1\n"
            "    for sh, st in zip(shape, stride):\n"
            "        if sh == 0:\n"
            "            continue\n"
            "        storage_n = max(storage_n, (sh - 1) * st + 1)\n"
            "    base = t.zeros(storage_n)\n"
            "    return t.as_strided(base, shape, stride).is_contiguous()\n"
            "\n"
            "cases = [\n"
            "    # (shape, stride, expected_label)\n"
            "    ((), (), True),\n"
            "    ((5,), (1,), True),\n"
            "    ((5,), (2,), False),\n"
            "    ((3, 4), (4, 1), True),\n"
            "    ((3, 4), (1, 3), False),                # column-major\n"
            "    ((2, 3, 4), (12, 4, 1), True),\n"
            "    ((2, 3, 4), (12, 1, 3), False),\n"
            "    ((2, 1, 4), (4, 999, 1), True),         # size-1 axis: stride ignored\n"
            "    ((2, 1, 4), (4, 0, 1), True),\n"
            "    ((4, 3), (3, 1, 99), False),            # length mismatch\n"
            "    ((0, 4), (4, 1), True),                 # any zero-size axis ⇒ True\n"
            "    ((3, 0), (4, 1), True),\n"
            "]\n"
            "\n"
            "for shape, stride, label in cases:\n"
            "    got = ex3_is_contiguous_from_meta(shape, stride)\n"
            "    assert got == label, (\n"
            "        f'shape={shape} stride={stride}: expected {label}, got {got}'\n"
            "    )\n"
            "    # Cross-check against torch ground truth where applicable.\n"
            "    if len(shape) == len(stride):\n"
            "        try:\n"
            "            gt = _gt(shape, stride)\n"
            "        except Exception:\n"
            "            gt = label   # skip cross-check if torch can't materialize\n"
            "        assert got == gt, (\n"
            "            f'shape={shape} stride={stride}: predicate disagrees with '\n"
            "            f'torch ground truth (predicate={got}, torch={gt}).'\n"
            "        )"
        ),
        "solution_body": (
            "def ex3_is_contiguous_from_meta(shape, stride):\n"
            "    if len(shape) != len(stride):\n"
            "        return False\n"
            "    if len(shape) == 0:\n"
            "        return True\n"
            "    if any(s == 0 for s in shape):\n"
            "        return True\n"
            "    expected_stride = 1\n"
            "    for k in range(len(shape) - 1, -1, -1):\n"
            "        if shape[k] != 1:\n"
            "            if stride[k] != expected_stride:\n"
            "                return False\n"
            "        expected_stride *= shape[k]\n"
            "    return True"
        ),
        "solution_notes": (
            "**Right-to-left walk.** The recurrence is `expected[k] = "
            "expected[k+1] * shape[k+1]`, so it's natural to start at the "
            "last axis (`expected = 1`) and accumulate as you step left.\n\n"
            "**Why size-1 axes are free.** Their stride is never multiplied "
            "by anything (the only valid index is 0). PyTorch optimizers "
            "exploit this to insert size-1 axes without losing contiguity — "
            "`x.unsqueeze(0).is_contiguous()` is `True` for any contiguous "
            "`x`.\n\n"
            "**Why zero-size axes are 'free'.** With an empty axis there's "
            "no memory to access, so contiguity is vacuous. This matches "
            "PyTorch's behavior (`t.zeros(0, 4).is_contiguous() is True`).\n\n"
            "**Reading this predicate is the same skill as predicting `view` "
            "errors.** If your tensor came from `transpose` and you can read "
            "off its strides, you instantly know whether the next `view` "
            "will crash."
        ),
        "extra_imports": [],
    },

    # ===================================================================
    # as-strided-windowing  —  ex3
    # NEW facet: STRIDED windows (step > 1) — gaps BETWEEN windows. ex1 +
    # ex2 both used unit-step (stride=(s,s) and stride=(s_B,s_IC,s_W,s_W)).
    # This drill uses stride=(s*step, s) — the missing 'step' parameter
    # for ARENA's strided-conv extension. Bloom: Apply.
    # ===================================================================
    {
        "atom_id": "as-strided-windowing",
        "subtopic": "PyTorch: as_strided windowing",
        "topic_folder": TOPIC_TENSOR_MECH,
        "atom_recap_md": RECAP_AS_STRIDED,
        "exercise_index": 3,
        "exercise_title": "strided 1-D windows with step greater than one",
        "slug": "strided-1d-windows-with-step-greater-than-one",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["sliding-window", "stride-step", "as_strided", "subsample"],
        "kcs": [
            "as-strided-window-stride",
            "as-strided-window-size",
        ],
        "lo": (
            "Apply `t.as_strided` with a multiplied outer stride to build a "
            "1-D sliding-window view whose windows step by `step` elements "
            "between consecutive positions (the missing-`stride>1` extension "
            "of ARENA's `conv1d_minimal`)."
        ),
        "prompt_body": (
            "Implement `ex3_strided_windows(x, K, step)`. Given a 1-D tensor "
            "`x` of length `L`, a window width `K`, and a positive integer "
            "`step`, return a zero-copy view of shape `(L_out, K)` where:\n\n"
            "- Row `i` is `x[i*step : i*step + K]`.\n"
            "- `L_out = (L - K) // step + 1`.\n\n"
            "Use `t.as_strided` with the right `size` and `stride`. Pull the "
            "element-stride of `x` from `x.stride()` — do **not** hard-code "
            "`1` (this matters if `x` was built via `x = source[::2]`).\n\n"
            "Required: `K >= 1`, `step >= 1`, `L >= K`. Return must alias "
            "`x`'s storage (no copy — writes propagate)."
        ),
        "stub": (
            "def ex3_strided_windows(x: Tensor, K: int, step: int) -> Tensor:\n"
            '    """Sliding windows of width K stepping by step elements."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Basic shape + value check.\n"
            "x = t.arange(10)\n"
            "w = ex3_strided_windows(x, K=3, step=2)\n"
            "assert tuple(w.shape) == (4, 3), f'expected (4,3), got {tuple(w.shape)}'\n"
            "expected = t.tensor([\n"
            "    [0, 1, 2],\n"
            "    [2, 3, 4],\n"
            "    [4, 5, 6],\n"
            "    [6, 7, 8],\n"
            "])\n"
            "assert t.equal(w, expected), f'value mismatch:\\n{w}\\nvs\\n{expected}'\n"
            "\n"
            "# Aliasing check — writes through w must propagate to x.\n"
            "x2 = t.arange(8).clone()\n"
            "w2 = ex3_strided_windows(x2, K=3, step=2)\n"
            "w2[0, 0] = -99\n"
            "assert x2[0].item() == -99, 'returned view must alias x storage'\n"
            "\n"
            "# step==1 collapses to the dense windowing of ex1.\n"
            "x3 = t.arange(6)\n"
            "w_dense = ex3_strided_windows(x3, K=2, step=1)\n"
            "assert tuple(w_dense.shape) == (5, 2)\n"
            "assert t.equal(w_dense, t.tensor([[0,1],[1,2],[2,3],[3,4],[4,5]]))\n"
            "\n"
            "# step==K → non-overlapping chunks.\n"
            "x4 = t.arange(12)\n"
            "chunks = ex3_strided_windows(x4, K=3, step=3)\n"
            "assert tuple(chunks.shape) == (4, 3)\n"
            "assert t.equal(chunks, t.tensor([[0,1,2],[3,4,5],[6,7,8],[9,10,11]]))\n"
            "\n"
            "# Non-unit-stride source — must use x.stride(0), not hard-coded 1.\n"
            "src = t.arange(20)\n"
            "sub = src[::2]                  # stride(0) == 2 elements, length 10\n"
            "w_sub = ex3_strided_windows(sub, K=3, step=2)\n"
            "assert tuple(w_sub.shape) == (4, 3)\n"
            "# Expected: sub == [0,2,4,6,8,10,12,14,16,18]; step-2 windows of width 3.\n"
            "exp_sub = t.tensor([[0,2,4],[4,6,8],[8,10,12],[12,14,16]])\n"
            "assert t.equal(w_sub, exp_sub), (\n"
            "    f'non-unit-stride source mishandled — did you hard-code stride=1?\\n'\n"
            "    f'got: {w_sub}\\nexpected: {exp_sub}'\n"
            ")"
        ),
        "solution_body": (
            "def ex3_strided_windows(x: Tensor, K: int, step: int) -> Tensor:\n"
            "    L = x.shape[0]\n"
            "    s, = x.stride()\n"
            "    L_out = (L - K) // step + 1\n"
            "    return t.as_strided(x, size=(L_out, K), stride=(s * step, s))"
        ),
        "solution_notes": (
            "**Outer stride scales by `step`, inner stride stays at `s`.** "
            "The outer stride says 'how to jump to the NEXT window'; the "
            "inner stride says 'how to walk inside one window'. The `step` "
            "argument only affects the outer.\n\n"
            "**`L_out` is the standard conv formula.** With dilation 1, "
            "`L_out = (L - K) // step + 1`. PyTorch's `nn.Conv1d` uses "
            "exactly this when you pass `stride=step`.\n\n"
            "**Why this is load-bearing for ARENA.** ARENA's `conv1d_minimal` "
            "drill assumes `stride=1`; the follow-up extension to "
            "`conv1d_general` requires this `step > 1` capability. Most "
            "learners get stuck at 'how do I represent a strided window'; "
            "the answer is exactly this two-stride trick."
        ),
        "extra_imports": [],
    },

    # ===================================================================
    # tensor-to-device  —  ex2
    # NEW facet: align a LIST of tensors to a common device (multi-tensor
    # helper) and verify which tensors moved vs. stayed. ex1 was single
    # tensor + cuda guard. Bloom: Apply.
    # ===================================================================
    {
        "atom_id": "tensor-to-device",
        "subtopic": "PyTorch: tensor.to(device)",
        "topic_folder": TOPIC_TENSOR_MECH,
        "atom_recap_md": RECAP_TENSOR_TO_DEVICE,
        "exercise_index": 2,
        "exercise_title": "align a list of tensors to a common device",
        "slug": "align-a-list-of-tensors-to-a-common-device",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["device", "alignment", "to", "idempotence"],
        "kcs": [
            "to-is-not-inplace",
            "pick-device-with-cuda-available",
        ],
        "lo": (
            "Apply `.to(device)` across a list of tensors so that all results "
            "live on the target device, exploiting `.to` idempotence so "
            "already-on-device tensors are returned as the SAME object."
        ),
        "prompt_body": (
            "Implement `ex2_align_to_device(tensors, device)`. Given a list "
            "of tensors (each possibly on a different device, possibly the "
            "same dtype or different) and a target `device` string (e.g. "
            "`'cpu'`), return a list of the same length where every tensor "
            "lives on `device`.\n\n"
            "**Hard requirements.**\n"
            "1. Use `.to(device)` on each tensor — do not write a manual "
            "copy or `clone()` fallback.\n"
            "2. **Idempotence**: if a tensor is ALREADY on `device`, "
            "`x.to(device)` must return the same object (`out is x` must "
            "be True). Don't add `clone()` after `.to()`.\n"
            "3. **Preserve order**: `out[i]` corresponds to `tensors[i]`.\n"
            "4. **No mutation**: do not modify the input list or its tensors.\n\n"
            "Inputs:\n"
            "- `tensors`: list of `Tensor`.\n"
            "- `device`: str, one of `'cpu'` or `'cuda'` or `'cuda:0'` etc.\n\n"
            "Output: list of `Tensor`, each on `device`."
        ),
        "stub": (
            "def ex2_align_to_device(tensors: list, device: str) -> list:\n"
            '    """Move every tensor in `tensors` to `device`."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Single-device suite (CPU-only) — the contract still applies.\n"
            "a = t.arange(3)\n"
            "b = t.zeros(2, 4)\n"
            "c = t.tensor([1.5, 2.5])\n"
            "out = ex2_align_to_device([a, b, c], 'cpu')\n"
            "assert isinstance(out, list) and len(out) == 3\n"
            "for i, (orig, new) in enumerate(zip([a, b, c], out)):\n"
            "    assert new.device.type == 'cpu', (\n"
            "        f'out[{i}] on {new.device}, expected cpu'\n"
            "    )\n"
            "    # Idempotence: already-on-cpu tensors must return the same object.\n"
            "    assert new is orig, (\n"
            "        f'out[{i}] is a NEW object — did you accidentally clone()? '\n"
            "        f'x.to(x.device) must return x itself.'\n"
            "    )\n"
            "\n"
            "# Empty list — degenerate but must not crash.\n"
            "assert ex2_align_to_device([], 'cpu') == []\n"
            "\n"
            "# Original list is not mutated.\n"
            "src = [t.arange(4), t.ones(2)]\n"
            "src_copy = list(src)\n"
            "_ = ex2_align_to_device(src, 'cpu')\n"
            "assert src == src_copy, 'do not mutate the input list (rebinding allowed; reordering not)'\n"
            "assert len(src) == 2\n"
            "\n"
            "# Order preserved.\n"
            "named = [t.tensor([float(i)]) for i in range(5)]\n"
            "moved = ex2_align_to_device(named, 'cpu')\n"
            "for i, m in enumerate(moved):\n"
            "    assert m.item() == float(i), f'order broken at {i}: got {m.item()}'"
        ),
        "solution_body": (
            "def ex2_align_to_device(tensors, device):\n"
            "    return [x.to(device) for x in tensors]"
        ),
        "solution_notes": (
            "**One-liner — and that's the whole point.** PyTorch's `.to()` "
            "already gives you idempotence for free: `x.to(x.device)` "
            "returns `x` (the exact same Python object). A list "
            "comprehension is enough.\n\n"
            "**Common over-engineering** — DON'T do this:\n"
            "```python\n"
            "return [x.to(device) if x.device != device else x.clone() for x in tensors]\n"
            "```\n"
            "The `clone()` allocates fresh storage you didn't ask for, "
            "breaking the idempotence contract and wasting memory in inner "
            "loops.\n\n"
            "**When CUDA is available**, the same code moves tensors across "
            "the PCI-e bus. The idempotence guarantee still holds: any "
            "tensor already on `cuda:0` is returned as-is (no D2D copy)."
        ),
        "extra_imports": [],
    },

    # ===================================================================
    # stack-vs-cat  —  ex2
    # NEW facet: BUILD an interleave op using stack + reshape (rank-up
    # then re-flatten). ex1 was 'classify which to use'. Now: USE stack
    # in the canonical interleave pattern. Bloom: Create.
    # ===================================================================
    {
        "atom_id": "stack-vs-cat",
        "subtopic": "PyTorch: stack vs cat",
        "topic_folder": TOPIC_TENSOR_MECH,
        "atom_recap_md": RECAP_STACK_VS_CAT,
        "exercise_index": 2,
        "exercise_title": "interleave two tensors with stack and reshape",
        "slug": "interleave-two-tensors-with-stack-and-reshape",
        "bloom_level": "Create",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["stack", "reshape", "interleave", "rank-change"],
        "kcs": [
            "stack-inserts-axis",
            "cat-along-existing-axis",
        ],
        "lo": (
            "Create an interleaving operation by composing `t.stack` (insert "
            "a pair axis) with `.reshape` (collapse it back) — building a "
            "rank-up-then-flatten pipeline rather than picking a single op."
        ),
        "prompt_body": (
            "Implement `ex2_interleave(a, b)`. Given two 1-D tensors `a`, `b` "
            "of equal length `n`, return the 1-D tensor of length `2n` whose "
            "values are `[a[0], b[0], a[1], b[1], ..., a[n-1], b[n-1]]`.\n\n"
            "**Required approach** (do NOT use a Python loop or `t.cat` "
            "alone):\n"
            "1. `t.stack([a, b], dim=1)` → shape `(n, 2)`. The new axis "
            "stores the (a, b) pair at each position `i`.\n"
            "2. `.reshape(-1)` (or `.flatten()`) → shape `(2n,)`. Row-major "
            "walk visits `(0,0), (0,1), (1,0), (1,1), ...` — which IS the "
            "interleaved order.\n\n"
            "Why this works: the row-major flatten visits the inner axis "
            "FASTEST, so the pair `(a[i], b[i])` is emitted together before "
            "moving to `i+1`.\n\n"
            "Inputs: two equal-length 1-D tensors of the same dtype.\n"
            "Output: 1-D tensor of length `2 * a.shape[0]`, same dtype."
        ),
        "stub": (
            "def ex2_interleave(a: Tensor, b: Tensor) -> Tensor:\n"
            '    """Interleave [a0,b0,a1,b1,...] via stack + reshape."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Tiny correctness check.\n"
            "a = t.tensor([1, 2, 3])\n"
            "b = t.tensor([10, 20, 30])\n"
            "out = ex2_interleave(a, b)\n"
            "assert tuple(out.shape) == (6,), f'expected (6,), got {tuple(out.shape)}'\n"
            "assert t.equal(out, t.tensor([1, 10, 2, 20, 3, 30])), (\n"
            "    f'order wrong: got {out}; expected [1,10,2,20,3,30].'\n"
            ")\n"
            "\n"
            "# dtype propagation.\n"
            "a = t.tensor([0.5, 1.5])\n"
            "b = t.tensor([0.0, 1.0])\n"
            "out = ex2_interleave(a, b)\n"
            "assert out.dtype == t.float32\n"
            "assert t.equal(out, t.tensor([0.5, 0.0, 1.5, 1.0]))\n"
            "\n"
            "# Length scaling.\n"
            "n = 100\n"
            "a = t.arange(n)\n"
            "b = t.arange(n) + 1000\n"
            "out = ex2_interleave(a, b)\n"
            "assert tuple(out.shape) == (2 * n,)\n"
            "# Even positions == a, odd positions == b.\n"
            "assert t.equal(out[0::2], a), 'even-position slice must equal a'\n"
            "assert t.equal(out[1::2], b), 'odd-position slice must equal b'\n"
            "\n"
            "# Edge case — length-0 inputs.\n"
            "empty_a = t.tensor([], dtype=t.float32)\n"
            "empty_b = t.tensor([], dtype=t.float32)\n"
            "out = ex2_interleave(empty_a, empty_b)\n"
            "assert tuple(out.shape) == (0,), f'empty-in must yield empty-out, got shape {tuple(out.shape)}'"
        ),
        "solution_body": (
            "def ex2_interleave(a: Tensor, b: Tensor) -> Tensor:\n"
            "    return t.stack([a, b], dim=1).reshape(-1)"
        ),
        "solution_notes": (
            "**Why `dim=1` (not `dim=0`).** Stack at `dim=0` would give "
            "shape `(2, n)`, and flattening walks ALL of `a` first then "
            "ALL of `b` — producing `[a0,a1,...,an-1,b0,b1,...]`, the "
            "WRONG order. Stack at `dim=1` puts the pair axis on the "
            "INSIDE, so it flattens fastest.\n\n"
            "**One-line composition.** This is the canonical stack+reshape "
            "idiom — it shows up in:\n"
            "- Audio stereo channel interleaving.\n"
            "- Bayer-pattern image deinterleaving (with `unfold`).\n"
            "- Skip-connection alignment when concatenating into a single "
            "buffer.\n\n"
            "**Why not `t.cat`.** `cat` only extends an existing axis. To "
            "interleave you must FIRST introduce a 'pair' axis (the job of "
            "`stack`) and then collapse it. There is no single-call cat "
            "that produces interleaved output."
        ),
        "extra_imports": [],
    },

    # ===================================================================
    # linalg-solve-batched  —  ex2
    # NEW facet: MULTIPLE right-hand sides — b shape (K, n, m) gives x
    # shape (K, n, m). ex1 was vector-RHS (K, n). Bloom: Apply.
    # ===================================================================
    {
        "atom_id": "linalg-solve-batched",
        "subtopic": "PyTorch: Batched linalg.solve",
        "topic_folder": TOPIC_GEOMETRY_CNN,
        "atom_recap_md": RECAP_LINALG_SOLVE,
        "exercise_index": 2,
        "exercise_title": "batched solve with multiple right-hand sides",
        "slug": "batched-solve-with-multiple-right-hand-sides",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["linalg", "solve", "multi-rhs", "inverse"],
        "kcs": [
            "linalg-solve-leading-batch",
            "linalg-solve-shape-contract",
        ],
        "lo": (
            "Apply `t.linalg.solve` with `A: (K, n, n)` and `b: (K, n, m)` "
            "to solve `K * m` linear systems sharing `K` LU factorizations, "
            "returning `x: (K, n, m)`."
        ),
        "prompt_body": (
            "Implement `ex2_solve_multi_rhs(A, B)`.\n\n"
            "- `A` has shape `(K, n, n)` — `K` square matrices.\n"
            "- `B` has shape `(K, n, m)` — `m` right-hand sides per matrix.\n"
            "- Return shape `(K, n, m)`: column `j` of slice `k` is the "
            "solution to `A[k] @ x == B[k, :, j]`.\n\n"
            "This is the shape contract `t.linalg.solve` already supports — "
            "pass `A` and `B` as-is, no per-column loop.\n\n"
            "**Why this is fast.** Each `A[k]` is LU-factorized ONCE; the "
            "`m` triangular solves share that factorization. If you looped "
            "over `j` you'd pay `m` extra LU factorizations per batch, "
            "trashing performance for `m >= 4`.\n\n"
            "Assume all `A[k]` are non-singular."
        ),
        "stub": (
            "def ex2_solve_multi_rhs(A: Tensor, B: Tensor) -> Tensor:\n"
            '    """Solve K batched n×n systems with m RHS columns each."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Hand-checked tiny case: K=2 systems, n=2, m=3 RHSs.\n"
            "A = t.tensor([\n"
            "    [[2.0, 0.0], [0.0, 3.0]],     # diagonal — easy to verify\n"
            "    [[1.0, 1.0], [0.0, 2.0]],\n"
            "])\n"
            "B = t.tensor([\n"
            "    [[2.0, 4.0, 6.0], [3.0, 6.0, 9.0]],   # A[0] solution = [1,2,3] / [1,2,3]\n"
            "    [[1.0, 0.0, 5.0], [2.0, 4.0, 2.0]],\n"
            "])\n"
            "X = ex2_solve_multi_rhs(A, B)\n"
            "assert tuple(X.shape) == (2, 2, 3), f'expected (2,2,3), got {tuple(X.shape)}'\n"
            "# Verify A @ X == B for every batch and column.\n"
            "AX = A @ X\n"
            "assert t.allclose(AX, B, atol=1e-5), (\n"
            "    f'A @ X != B:\\nAX = {AX}\\nB = {B}'\n"
            ")\n"
            "\n"
            "# Special case — solve A x = I gives A^{-1}.\n"
            "rng = t.Generator().manual_seed(7)\n"
            "K, n = 4, 3\n"
            "Abig = t.randn(K, n, n, generator=rng)\n"
            "# Nudge away from singular by adding scaled identity.\n"
            "Abig = Abig + 2.0 * t.eye(n).expand(K, n, n)\n"
            "I = t.eye(n).expand(K, n, n).contiguous()\n"
            "Ainv = ex2_solve_multi_rhs(Abig, I)\n"
            "assert tuple(Ainv.shape) == (K, n, n)\n"
            "should_be_I = Abig @ Ainv\n"
            "assert t.allclose(should_be_I, I, atol=1e-4), (\n"
            "    f'A @ A^-1 != I:\\n{should_be_I}'\n"
            ")\n"
            "\n"
            "# Vector-RHS (m=1) sanity: shape (K, n, 1) round-trips correctly.\n"
            "B1 = t.randn(K, n, 1, generator=rng)\n"
            "X1 = ex2_solve_multi_rhs(Abig, B1)\n"
            "assert tuple(X1.shape) == (K, n, 1)\n"
            "assert t.allclose(Abig @ X1, B1, atol=1e-4)"
        ),
        "solution_body": (
            "def ex2_solve_multi_rhs(A: Tensor, B: Tensor) -> Tensor:\n"
            "    return t.linalg.solve(A, B)"
        ),
        "solution_notes": (
            "**One call — that's the whole answer.** The shape contract of "
            "`t.linalg.solve` already handles trailing `m` columns: when "
            "`B.shape == (..., n, m)`, the result is `(..., n, m)`.\n\n"
            "**Why ARENA cares about this shape.** The triangle-rasterization "
            "drill solves a barycentric system per (triangle, pixel) pair. "
            "With `K` triangles and `m` query pixels, the natural batch is "
            "`A: (K, 2, 2)` and `B: (K, 2, m)`, returning `(K, 2, m)`. One "
            "`solve` call replaces a nested Python loop.\n\n"
            "**Inverse via solve.** `A @ A^-1 = I` ⇒ `A^-1 = solve(A, I)`. "
            "This is the recommended way to compute inverses in PyTorch — "
            "explicit `t.linalg.inv` is implemented as `solve(A, I)` "
            "internally, and is more numerically stable than `1/A` or "
            "Gauss-Jordan elimination written by hand."
        ),
        "extra_imports": [],
    },

    # ===================================================================
    # conv-output-shape  —  ex2
    # NEW facet: INVERT the formula — given desired output shape + kernel
    # + stride, compute padding needed for 'SAME'-style output. ex1 was
    # forward-direction. Bloom: Analyze.
    # ===================================================================
    {
        "atom_id": "conv-output-shape",
        "subtopic": "CNN: Conv output shape",
        "topic_folder": TOPIC_GEOMETRY_CNN,
        "atom_recap_md": RECAP_CONV_OUTPUT_SHAPE,
        "exercise_index": 2,
        "exercise_title": "invert the conv formula to compute SAME-style padding",
        "slug": "invert-the-conv-formula-to-compute-same-style-padding",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["conv", "same-padding", "inverse-formula", "shape"],
        "kcs": [
            "conv-output-shape-formula",
            "conv-shape-batch-pass-through",
        ],
        "lo": (
            "Analyze the conv output-shape formula in reverse: given desired "
            "`L_out`, kernel `K`, and stride `S`, compute the (symmetric) "
            "padding `P` needed so that a `Conv1d` produces exactly `L_out`."
        ),
        "prompt_body": (
            "Implement `ex2_same_padding(L_in, L_out, K, S)`. Solve the "
            "forward formula\n"
            "```\n"
            "L_out = (L_in + 2*P - K) // S + 1\n"
            "```\n"
            "for `P`, returning the smallest non-negative integer that "
            "achieves *at least* `L_out`. Specifically:\n\n"
            "1. Algebra: drop the floor first → "
            "`P_real = ((L_out - 1) * S - L_in + K) / 2`.\n"
            "2. Round UP: `P = max(0, ceil(P_real))`.\n"
            "3. Verify by re-running the forward formula with that `P` — "
            "the actual output length must be `>= L_out` and within `S` of "
            "it. Return `P`.\n\n"
            "Then write a quick sanity check inside your function: assert "
            "`(L_in + 2*P - K) // S + 1 >= L_out`. The test will probe the "
            "same identity from the outside.\n\n"
            "Inputs (all positive ints): `L_in, L_out, K, S`.\n"
            "Output: integer padding `P >= 0`."
        ),
        "stub": (
            "def ex2_same_padding(L_in: int, L_out: int, K: int, S: int) -> int:\n"
            '    """Smallest P that makes Conv1d produce >= L_out from L_in."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import math\n"
            "\n"
            "def _forward(L_in, K, S, P):\n"
            "    return (L_in + 2 * P - K) // S + 1\n"
            "\n"
            "# Classic SAME with stride 1, odd K: P = (K - 1) // 2.\n"
            "for K_ in [1, 3, 5, 7, 9]:\n"
            "    P = ex2_same_padding(L_in=32, L_out=32, K=K_, S=1)\n"
            "    assert P == (K_ - 1) // 2, (\n"
            "        f'odd K stride-1 SAME: K={K_} → expected P={(K_-1)//2}, got {P}'\n"
            "    )\n"
            "    assert _forward(32, K_, 1, P) >= 32\n"
            "\n"
            "# Stride > 1 halving — common ARENA setup.\n"
            "# L_in=32, S=2, K=3 → desired L_out=16 needs P=1: (32 + 2 - 3)//2 + 1 = 16.\n"
            "P = ex2_same_padding(L_in=32, L_out=16, K=3, S=2)\n"
            "assert _forward(32, 3, 2, P) >= 16\n"
            "assert _forward(32, 3, 2, P) - 16 < 2   # within one stride of target\n"
            "\n"
            "# When no padding is needed, function returns 0.\n"
            "P = ex2_same_padding(L_in=10, L_out=8, K=3, S=1)\n"
            "assert P == 0, f'no padding needed, got {P}'\n"
            "assert _forward(10, 3, 1, P) == 8\n"
            "\n"
            "# Sweep random configurations and verify the inverse identity.\n"
            "import random\n"
            "rng = random.Random(0)\n"
            "for _ in range(50):\n"
            "    K_ = rng.choice([1, 2, 3, 4, 5])\n"
            "    S_ = rng.choice([1, 2, 3])\n"
            "    L_in = rng.randint(K_, 64)\n"
            "    L_out = rng.randint(1, L_in)\n"
            "    P = ex2_same_padding(L_in, L_out, K_, S_)\n"
            "    assert P >= 0, f'P must be >=0, got {P}'\n"
            "    actual = _forward(L_in, K_, S_, P)\n"
            "    assert actual >= L_out, (\n"
            "        f'L_in={L_in} L_out={L_out} K={K_} S={S_} P={P}: '\n"
            "        f'forward gives {actual} < target {L_out}.'\n"
            "    )\n"
            "    # Tightness: P-1 (when > 0) should under-shoot.\n"
            "    if P > 0:\n"
            "        actual_minus = _forward(L_in, K_, S_, P - 1)\n"
            "        assert actual_minus < L_out, (\n"
            "            f'L_in={L_in} L_out={L_out} K={K_} S={S_}: '\n"
            "            f'P={P} is not the smallest — P-1={P-1} also works (gave {actual_minus}).'\n"
            "        )"
        ),
        "solution_body": (
            "def ex2_same_padding(L_in, L_out, K, S):\n"
            "    import math\n"
            "    P_real = ((L_out - 1) * S - L_in + K) / 2.0\n"
            "    P = max(0, math.ceil(P_real))\n"
            "    # Sanity check: forward formula must achieve >= L_out.\n"
            "    assert (L_in + 2 * P - K) // S + 1 >= L_out\n"
            "    return P"
        ),
        "solution_notes": (
            "**Algebra in one line.** Drop the floor (cast to real), then "
            "round up to the nearest integer. The floor in the FORWARD "
            "formula means the forward output rounds DOWN, so to guarantee "
            "we reach `L_out` we must round the inverse UP.\n\n"
            "**Why `ceil` and not `round`.** Banker's rounding (which "
            "Python's `round` does at `.5`) would sometimes give `P` that's "
            "ONE too small — the forward formula then under-shoots `L_out` "
            "by exactly 1. `ceil` is the safe direction.\n\n"
            "**ARENA practical use.** When you build a CNN-from-scratch "
            "downsampler that halves the spatial extent at every stage, "
            "you call `ex2_same_padding(L_in=in_size, L_out=in_size//2, "
            "K=kernel, S=2)` at each layer. Without this inverse you'd "
            "guess-and-check the padding manually."
        ),
        "extra_imports": [],
    },

    # ===================================================================
    # conv-padding-zero  —  ex2
    # NEW facet: 2-D ASYMMETRIC padding (top/bottom/left/right differ).
    # ex1 was symmetric 1-D (left, right). Bloom: Apply.
    # ===================================================================
    {
        "atom_id": "conv-padding-zero",
        "subtopic": "CNN: Conv zero padding",
        "topic_folder": TOPIC_GEOMETRY_CNN,
        "atom_recap_md": RECAP_CONV_PADDING,
        "exercise_index": 2,
        "exercise_title": "asymmetric 2-D zero padding by slice assignment",
        "slug": "asymmetric-2d-zero-padding-by-slice-assignment",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["padding", "2d", "asymmetric", "slice-assign"],
        "kcs": [
            "pad-allocate-zero-buffer",
            "pad-slice-assign-interior",
        ],
        "lo": (
            "Apply the allocate-zero-buffer-then-assign-interior padding "
            "pattern to a 2-D input with four independent side amounts "
            "(top, bottom, left, right) producing a `(B, IC, top+H+bottom, "
            "left+W+right)` output."
        ),
        "prompt_body": (
            "Implement `ex2_pad2d_asymmetric(x, top, bottom, left, right)`.\n\n"
            "- `x` has shape `(B, IC, H, W)`.\n"
            "- `top, bottom, left, right` are non-negative ints (possibly "
            "different).\n"
            "- Return shape `(B, IC, top + H + bottom, left + W + right)`.\n"
            "  - The first `top` rows and last `bottom` rows are exactly "
            "zero.\n"
            "  - The first `left` columns and last `right` columns are "
            "exactly zero.\n"
            "  - The interior `[..., top:top+H, left:left+W]` equals `x`.\n\n"
            "**Required approach** (do not use `F.pad`):\n"
            "1. `out = x.new_zeros(B, IC, top + H + bottom, left + W + right)`.\n"
            "2. `out[..., top:top + H, left:left + W] = x`.\n"
            "3. Return `out`.\n\n"
            "Use `new_zeros` so dtype and device inherit from `x`. The "
            "interior slice assignment is the single load-bearing step."
        ),
        "stub": (
            "def ex2_pad2d_asymmetric(x: Tensor, top: int, bottom: int,\n"
            "                         left: int, right: int) -> Tensor:\n"
            '    """Zero-pad x with four independent side amounts."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import torch.nn.functional as F\n"
            "\n"
            "# Tiny correctness check.\n"
            "x = t.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])   # (1, 1, 2, 2)\n"
            "out = ex2_pad2d_asymmetric(x, top=1, bottom=2, left=3, right=0)\n"
            "expected_shape = (1, 1, 1 + 2 + 2, 3 + 2 + 0)\n"
            "assert tuple(out.shape) == expected_shape, (\n"
            "    f'expected shape {expected_shape}, got {tuple(out.shape)}'\n"
            ")\n"
            "# Interior must equal x.\n"
            "assert t.equal(out[..., 1:1+2, 3:3+2], x), 'interior must equal x'\n"
            "# All padded entries must be exactly zero.\n"
            "mask = t.ones_like(out, dtype=t.bool)\n"
            "mask[..., 1:1+2, 3:3+2] = False\n"
            "assert (out[mask] == 0).all(), 'padded region must be exactly zero'\n"
            "\n"
            "# Symmetric case must match nn.functional.pad (which uses\n"
            "# argument order: last-axis pad first → (left, right, top, bottom)).\n"
            "rng = t.Generator().manual_seed(2)\n"
            "x2 = t.randn(2, 3, 5, 7, generator=rng)\n"
            "got = ex2_pad2d_asymmetric(x2, top=2, bottom=2, left=4, right=4)\n"
            "want = F.pad(x2, (4, 4, 2, 2), mode='constant', value=0)\n"
            "assert t.equal(got, want), 'symmetric pad disagrees with F.pad'\n"
            "\n"
            "# Asymmetric case must also match F.pad's 4-tuple.\n"
            "got2 = ex2_pad2d_asymmetric(x2, top=1, bottom=3, left=2, right=5)\n"
            "want2 = F.pad(x2, (2, 5, 1, 3), mode='constant', value=0)\n"
            "assert t.equal(got2, want2), 'asymmetric pad disagrees with F.pad'\n"
            "\n"
            "# All-zero pad amounts → identity (no extra zeros).\n"
            "got3 = ex2_pad2d_asymmetric(x2, 0, 0, 0, 0)\n"
            "assert tuple(got3.shape) == tuple(x2.shape)\n"
            "assert t.equal(got3, x2)\n"
            "\n"
            "# dtype/device inheritance — int dtype must round-trip.\n"
            "xi = t.arange(12).reshape(1, 1, 3, 4)\n"
            "got4 = ex2_pad2d_asymmetric(xi, top=1, bottom=0, left=0, right=2)\n"
            "assert got4.dtype == xi.dtype, (\n"
            "    f'dtype not inherited: got {got4.dtype}, expected {xi.dtype}; '\n"
            "    f'did you use t.zeros instead of x.new_zeros?'\n"
            ")"
        ),
        "solution_body": (
            "def ex2_pad2d_asymmetric(x: Tensor, top: int, bottom: int,\n"
            "                         left: int, right: int) -> Tensor:\n"
            "    B, IC, H, W = x.shape\n"
            "    out = x.new_zeros(B, IC, top + H + bottom, left + W + right)\n"
            "    out[..., top:top + H, left:left + W] = x\n"
            "    return out"
        ),
        "solution_notes": (
            "**Four independent side amounts** is the generic case; "
            "PyTorch's `nn.Conv2d(padding=k)` shorthand collapses it to "
            "the symmetric `top=bottom=left=right=k`. The full 4-tuple is "
            "what you need for 'SAME' padding when the conv stride or "
            "kernel size doesn't split evenly.\n\n"
            "**Argument-order trap.** `F.pad`'s tuple is in REVERSE axis "
            "order: `(left, right, top, bottom)` for a 2-D pad. Our "
            "function uses natural order (`top, bottom, left, right`) — "
            "exposing the trap by making both orderings appear in the "
            "test side-by-side. Remembering 'F.pad walks axes inside-out' "
            "(last axis first) is the single fact that prevents most "
            "padding bugs.\n\n"
            "**`x.new_zeros` over `t.zeros`.** The former inherits dtype "
            "and device automatically. The test for integer round-trip "
            "catches the difference: `t.zeros(...)` defaults to `float32` "
            "and would silently demote your int input."
        ),
        "extra_imports": [],
    },
]


def _verify_all(specs):
    import torch as t
    import numpy as np
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

        # Some test_bodies reference an exception-raising decoy (train_swapped).
        # Those bodies are defined inside the STUB (not solution), so we must
        # also exec the stub to install the helper. The stub has a
        # NotImplementedError-only function (e.g. ex3_train_fixed) plus possibly
        # a real helper above it. Exec stub first, then OVERWRITE the target
        # function via solution_body.
        try:
            exec(spec["stub"], ns)
        except Exception:
            # Stub may include unbound names if it's purely a raise — that's
            # caught by the exec, but only at call time. We tolerate name-level
            # failures here by retrying without the stub.
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
    print(f"[deepening_a_batch7] Verifying {len(SPECS)} specs...")
    _verify_all(SPECS)

    print(f"\n[deepening_a_batch7] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_a_batch7] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
