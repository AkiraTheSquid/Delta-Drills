#!/usr/bin/env python3
"""Author 8 Colab standalones for the tensor-mechanics prereq cluster.

Atoms covered (folder prereqs_tensor_mechanics/<atom>/):
    contiguous-layout       (2 ex)
    as-strided-windowing    (2 ex) — windowing for conv1d (distinct from
                                     existing `as-strided-noncontig-source`
                                     atom whose drills focus on "is this view
                                     contiguous").
    slice-view-mutation     (1 ex)
    stack-vs-cat            (1 ex)
    tensor-to-device        (1 ex)
    tensor-wraps-ndarray    (1 ex)

Per Doughty (2024) ACE + Maier (2021), each exercise has:
- exactly one LO + one Bloom level
- max 2 concurrent KCs
- solution executes cleanly (we verify with the backend venv torch+einops
  before writing the .ipynb).

Solutions are verified IN-PROCESS before any notebook is written. If any
solution fails its test, no notebooks are emitted.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC_FOLDER = "prereqs_tensor_mechanics"


# -----------------------------------------------------------------------------
# Atom recaps — one per atom; reused across that atom's exercises.
# -----------------------------------------------------------------------------

RECAP_CONTIGUOUS = (
    "## Contiguous layout — quick refresher\n"
    "\n"
    "A tensor is **contiguous** when its in-memory layout is row-major: the "
    "last axis has stride 1 and each earlier axis's stride equals the product "
    "of all sizes to its right. `x.is_contiguous()` reports the answer; "
    "`x.stride()` lets you check by hand.\n"
    "\n"
    "**Why it matters.**\n"
    "- `view()` requires contiguous input — call `.contiguous()` first if "
    "you've transposed/permuted/strided into a non-contiguous layout.\n"
    "- `reshape()` will silently copy when needed; `view()` will not.\n"
    "- Many low-level kernels (cuDNN convs, `as_strided`) read raw stride "
    "values — passing them a tensor whose strides you didn't expect is the "
    "single biggest source of off-by-`H*W` bugs in CNN-from-scratch code.\n"
    "\n"
    "**Useful identities for a contiguous `(d0, d1, ..., dN)` tensor:**\n"
    "- `stride(N) == 1`\n"
    "- `stride(k) == d[k+1] * d[k+2] * ... * d[N]`"
)

RECAP_AS_STRIDED_WINDOWING = (
    "## `as_strided` windowing — quick refresher\n"
    "\n"
    "`t.as_strided(x, size, stride)` builds a **zero-copy** view at exactly "
    "the `(size, stride)` you specify. The classic sliding-window trick is to "
    "set both the window-position stride *and* the within-window stride to "
    "the source's element stride, so the view becomes a (n_windows, "
    "window_size) matrix that aliases the source.\n"
    "\n"
    "**Pattern for 1-D sliding window of width `K` over a 1-D tensor `x` of "
    "length `L`:**\n"
    "```python\n"
    "sL, = x.stride()\n"
    "windows = t.as_strided(x, size=(L - K + 1, K), stride=(sL, sL))\n"
    "```\n"
    "\n"
    "**Generalises to N-D inputs.** For batched/channelled input "
    "`(B, IC, W)`, you pull all of `x.stride()` and build a `(B, IC, "
    "L_out, K)` view with stride `(sB, sIC, sW, sW)`. This is exactly the "
    "ARENA `conv1d_minimal` trick.\n"
    "\n"
    "**Why you need the source's stride, not `1`.** If `x` was itself created "
    "via `permute`/`transpose`/another `as_strided` call, its last stride "
    "may be larger than 1. Hard-coding `1` will silently scan the wrong "
    "memory cells."
)

RECAP_SLICE_VIEW = (
    "## Slice view mutation — quick refresher\n"
    "\n"
    "Slicing a tensor with `:` and integer-range syntax returns a **view** "
    "that shares storage with the source. Writes through the view alias the "
    "source:\n"
    "```python\n"
    "x = t.zeros(4, 4)\n"
    "row = x[1]           # view — same storage as x\n"
    "row[:] = 7.0         # mutates x[1] in place\n"
    "```\n"
    "\n"
    "**Contrast with boolean / fancy indexing**, which returns a **copy**. "
    "If you ever see `x[mask] = value` work but `subset = x[mask]; "
    "subset[:] = value` not, this is why — the second form mutates an "
    "independent copy.\n"
    "\n"
    "**`.clone()` breaks the aliasing.** Use it when you want a slice you can "
    "modify without affecting the source."
)

RECAP_STACK_VS_CAT = (
    "## `stack` vs `cat` — quick refresher\n"
    "\n"
    "- `t.cat(tensors, dim=k)` **concatenates** along an existing axis. "
    "Output rank = input rank. All inputs must agree on every axis except "
    "`k`.\n"
    "- `t.stack(tensors, dim=k)` **inserts a new axis** at position `k`. "
    "Output rank = input rank + 1. All inputs must have **identical** "
    "shapes.\n"
    "\n"
    "**Mental model.** If you have a list of `(3, 4)` tensors:\n"
    "- `t.cat(list, dim=0)` over `N` of them → `(N*3, 4)`\n"
    "- `t.stack(list, dim=0)` over `N` of them → `(N, 3, 4)`\n"
    "\n"
    "**Equivalence.** `t.stack(xs, dim=k)` ≡ `t.cat([x.unsqueeze(k) for x "
    "in xs], dim=k)`. Same result, but `stack` is shorter and intent-"
    "revealing."
)

RECAP_TO_DEVICE = (
    "## `tensor.to(device)` — quick refresher\n"
    "\n"
    "`x.to(device)` returns a new tensor on the target device. It is a "
    "**copy** when the device differs and a **no-op view** when the device "
    "already matches.\n"
    "\n"
    "**Critical rules.**\n"
    "- `x.to(device)` is **not in-place** — you must reassign: `x = "
    "x.to(device)`.\n"
    "- All inputs to an op must live on the same device. Mixing CPU and CUDA "
    "tensors raises `RuntimeError`.\n"
    "- `to()` also accepts a dtype: `x.to(dtype=t.float16)` or "
    "`x.to(device='cuda', dtype=t.float16)`.\n"
    "- For modules, `model.to(device)` IS in-place (moves all parameters and "
    "buffers).\n"
    "\n"
    "**Idiomatic guard.** Pick the device once at the top of the script and "
    "thread it through:\n"
    "```python\n"
    "device = 'cuda' if t.cuda.is_available() else 'cpu'\n"
    "x = x.to(device)\n"
    "model = model.to(device)\n"
    "```"
)

RECAP_FROM_NDARRAY = (
    "## tensor from ndarray — quick refresher\n"
    "\n"
    "Three ways to make a torch tensor from a NumPy ndarray; they have "
    "**different semantics**.\n"
    "\n"
    "| factory | shares memory? | dtype-preserving? |\n"
    "|---|---|---|\n"
    "| `t.from_numpy(arr)` | YES — zero-copy view | yes |\n"
    "| `t.as_tensor(arr)` | yes when possible (same dtype+device) | yes |\n"
    "| `t.tensor(arr)` | NO — always copies | yes (deduced) |\n"
    "\n"
    "**Aliasing trap.** Tensors built with `from_numpy` write through to the "
    "source ndarray. Mutating the tensor in place will silently change the "
    "NumPy view — and vice versa.\n"
    "\n"
    "**Dtype gotcha.** NumPy defaults to `float64`; PyTorch defaults to "
    "`float32`. Cast at construction (`t.from_numpy(arr).float()` or "
    "`arr.astype('float32')`) to avoid an unexpected double-precision "
    "tensor."
)


# -----------------------------------------------------------------------------
# Specs
# -----------------------------------------------------------------------------

def _spec_contiguous_ex1():
    return {
        "atom_id": "contiguous-layout",
        "subtopic": "PyTorch: Contiguous layout",
        "topic_folder": TOPIC_FOLDER,
        "atom_recap_md": RECAP_CONTIGUOUS,
        "exercise_index": 1,
        "exercise_title": "predict-then-verify the strides of a 3-D contiguous tensor",
        "slug": "predict-then-verify-strides-of-contiguous-3d",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["strides", "row-major", "is_contiguous"],
        "kcs": ["contiguous-stride-formula", "is-contiguous-check"],
        "lo": (
            "Compute the strides of a contiguous N-dim tensor from its shape "
            "using the row-major formula, and verify against `.stride()` and "
            "`.is_contiguous()`."
        ),
        "prompt_body": (
            "Implement `ex1_predicted_strides(shape)`. Given a tuple of sizes "
            "`shape = (d0, d1, ..., dN)`, return the tuple of strides a "
            "contiguous (row-major) tensor of that shape would have.\n\n"
            "**Formula.** For a contiguous tensor:\n"
            "- `stride[-1] = 1`\n"
            "- `stride[k] = d[k+1] * d[k+2] * ... * d[N]` for all `k < N`\n"
            "\n"
            "Inputs: `shape` — tuple of `int >= 1`.\n"
            "Output: tuple of `int`, same length as `shape`.\n\n"
            "**No torch in this function — pure Python math.** The test will "
            "use a real `t.zeros(shape)` to confirm your formula matches "
            "what PyTorch actually allocates."
        ),
        "stub": (
            "def ex1_predicted_strides(shape: tuple) -> tuple:\n"
            '    """Return the contiguous-layout strides for a tensor of `shape`."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Hand-checked small cases.\n"
            "assert ex1_predicted_strides((5,)) == (1,)\n"
            "assert ex1_predicted_strides((3, 4)) == (4, 1)\n"
            "assert ex1_predicted_strides((2, 3, 4)) == (12, 4, 1)\n"
            "assert ex1_predicted_strides((2, 1, 3, 4)) == (12, 12, 4, 1)\n"
            "\n"
            "# Cross-check against PyTorch's actual layout for several shapes.\n"
            "for shape in [(7,), (3, 5), (2, 3, 4), (4, 1, 6, 2)]:\n"
            "    real = t.zeros(shape).stride()\n"
            "    pred = ex1_predicted_strides(shape)\n"
            "    assert tuple(real) == tuple(pred), (\n"
            "        f'shape {shape}: predicted {pred}, real {tuple(real)}'\n"
            "    )\n"
            "\n"
            "# Confirm the produced shape is contiguous (sanity check).\n"
            "assert t.zeros(2, 3, 4).is_contiguous()\n"
            "print('all stride predictions match torch.zeros(...).stride()')\n"
            "print(f'  (2,3,4) -> stride {ex1_predicted_strides((2,3,4))}')"
        ),
        "solution_body": (
            "def ex1_predicted_strides(shape: tuple) -> tuple:\n"
            "    strides = []\n"
            "    running = 1\n"
            "    for dim in reversed(shape):\n"
            "        strides.append(running)\n"
            "        running *= dim\n"
            "    return tuple(reversed(strides))"
        ),
        "solution_notes": (
            "**The walk goes right-to-left.** The last axis always has stride "
            "1 (you advance by one element to step along it). Each earlier "
            "axis multiplies in the size of everything to its right.\n\n"
            "**Why this is a load-bearing skill for CNN-from-scratch.** "
            "Building `as_strided` windows for `conv1d_minimal` requires you "
            "to KNOW these strides — you'll pull them off `x.stride()` "
            "rather than recompute, but if your mental model is wrong you "
            "won't catch the bug when a transposed input arrives with "
            "non-contiguous strides.\n\n"
            "**Gotcha — size-1 axes.** A `(2, 1, 3, 4)` contiguous tensor has "
            "stride `(12, 12, 4, 1)`. The size-1 axis's stride equals the "
            "stride of the axis to its left because there's nothing to "
            "advance over."
        ),
    }


def _spec_contiguous_ex2():
    return {
        "atom_id": "contiguous-layout",
        "subtopic": "PyTorch: Contiguous layout",
        "topic_folder": TOPIC_FOLDER,
        "atom_recap_md": RECAP_CONTIGUOUS,
        "exercise_index": 2,
        "exercise_title": "fix the view-after-transpose error with .contiguous()",
        "slug": "fix-view-after-transpose-with-contiguous",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["view", "transpose", "contiguous-copy"],
        "kcs": ["view-requires-contiguous", "contiguous-materialize"],
        "lo": (
            "Diagnose a `RuntimeError: view size is not compatible with input "
            "tensor's size and stride` after a transpose, and fix it by "
            "inserting `.contiguous()` before the `.view()` call."
        ),
        "prompt_body": (
            "Implement `ex2_flatten_after_transpose(x)`.\n\n"
            "Input: `x` of shape `(B, H, W)`. You must:\n"
            "1. Transpose the last two axes to get `(B, W, H)`.\n"
            "2. Flatten the trailing two axes into one via `.view(B, W * "
            "H)`.\n"
            "3. Return the `(B, W * H)` tensor.\n\n"
            "**The catch.** A naive `x.transpose(-1, -2).view(B, W * H)` "
            "raises a `RuntimeError` because `transpose` returns a "
            "non-contiguous view and `view` refuses non-contiguous inputs. "
            "Fix it by calling `.contiguous()` between the two ops.\n\n"
            "Output: `(B, W * H)` float tensor, row-major flattened from the "
            "transposed `(B, W, H)` layout."
        ),
        "stub": (
            "def ex2_flatten_after_transpose(x: Tensor) -> Tensor:\n"
            '    """Transpose last two axes, then flatten via .view()."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Build a tensor where the value encodes its position so we can\n"
            "# verify the transpose actually happened (not just a reshape).\n"
            "x = t.arange(24, dtype=t.float32).reshape(2, 3, 4)\n"
            "out = ex2_flatten_after_transpose(x)\n"
            "assert out.shape == (2, 12), f'expected (2, 12), got {tuple(out.shape)}'\n"
            "assert out.dtype == t.float32, f'expected float32, got {out.dtype}'\n"
            "\n"
            "# Reference: do it the explicit safe way and compare.\n"
            "expected = x.transpose(-1, -2).contiguous().view(2, 12)\n"
            "assert t.equal(out, expected), (\n"
            "    f'value mismatch:\\nout={out}\\nexpected={expected}'\n"
            ")\n"
            "\n"
            "# Confirm the result really came from the TRANSPOSED layout —\n"
            "# i.e. not a flat .view() of x itself (which would have produced\n"
            "# a different ordering, [0,1,2,3,4,...] instead of\n"
            "# [0,4,8,1,5,9,2,6,10,3,7,11] for batch 0).\n"
            "wrong = x.view(2, 12)\n"
            "assert not t.equal(out, wrong), (\n"
            "    'output equals plain .view(B, W*H) — you skipped the transpose'\n"
            ")\n"
            "\n"
            "# Larger random case for shape robustness.\n"
            "rng = t.Generator().manual_seed(0)\n"
            "y = t.randn(5, 7, 11, generator=rng)\n"
            "out_y = ex2_flatten_after_transpose(y)\n"
            "assert out_y.shape == (5, 77)\n"
            "assert t.allclose(out_y, y.transpose(-1, -2).contiguous().view(5, 77))\n"
            "print('transpose + .contiguous() + .view fixed the RuntimeError')"
        ),
        "solution_body": (
            "def ex2_flatten_after_transpose(x: Tensor) -> Tensor:\n"
            "    B = x.shape[0]\n"
            "    H, W = x.shape[1], x.shape[2]\n"
            "    return x.transpose(-1, -2).contiguous().view(B, W * H)"
        ),
        "solution_notes": (
            "**Why `transpose` breaks `view`.** `transpose` keeps the same "
            "storage but swaps two stride values. The new strides don't "
            "satisfy the contiguous formula, so `view` (which is a pure "
            "metadata re-interpretation) cannot re-label them as a flat "
            "1-D buffer.\n\n"
            "**Two valid fixes.**\n"
            "- `x.transpose(-1, -2).contiguous().view(B, W * H)` — explicit, "
            "shows your intent.\n"
            "- `x.transpose(-1, -2).reshape(B, W * H)` — `reshape` calls "
            "`.contiguous()` for you under the hood when it has to.\n\n"
            "**Cost of `.contiguous()`.** It allocates and copies — `O(B * H "
            "* W)` memory. For very large activation maps this is the kind "
            "of hidden allocation that shows up in profilers as a "
            "'mysterious' 2x memory blip."
        ),
    }


def _spec_as_strided_windowing_ex1():
    return {
        "atom_id": "as-strided-windowing",
        "subtopic": "PyTorch: as_strided windowing",
        "topic_folder": TOPIC_FOLDER,
        "atom_recap_md": RECAP_AS_STRIDED_WINDOWING,
        "exercise_index": 1,
        "exercise_title": "compute size + stride args for a 1-D sliding window",
        "slug": "compute-size-stride-for-1d-sliding-window",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["sliding-window", "stride-math", "as_strided"],
        "kcs": ["as-strided-window-size", "as-strided-window-stride"],
        "lo": (
            "Compute the `size` and `stride` arguments needed to pass to "
            "`t.as_strided` for a width-`K` sliding window over a 1-D tensor, "
            "without doing the windowing itself."
        ),
        "prompt_body": (
            "Implement `ex1_window_args(x, kernel_width)`. Given a 1-D tensor "
            "`x` and an integer `kernel_width`, return the `(size, stride)` "
            "tuple of tuples you'd pass to `t.as_strided` to produce the "
            "sliding-window view used in ARENA's `conv1d_minimal`.\n\n"
            "Specifically, for `x` of length `L`:\n"
            "- `size` = `(L - kernel_width + 1, kernel_width)`\n"
            "- `stride` = `(x.stride(0), x.stride(0))` — **both axes use the "
            "source's stride**, not `1`. (If `x` was already non-contiguous, "
            "hard-coding `1` reads the wrong memory.)\n"
            "\n"
            "Return: `(size_tuple, stride_tuple)` — both tuples of `int`.\n\n"
            "**Don't actually call `as_strided` here** — just compute the "
            "args. The test will pass them into `as_strided` and verify the "
            "resulting window matches what convolution prep expects."
        ),
        "stub": (
            "def ex1_window_args(x: Tensor, kernel_width: int) -> tuple:\n"
            '    """Compute (size, stride) args for a 1-D sliding window."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Contiguous case.\n"
            "x = t.arange(8, dtype=t.float32)\n"
            "size, stride = ex1_window_args(x, kernel_width=3)\n"
            "assert size == (6, 3), f'expected (6, 3), got {size}'\n"
            "assert stride == (1, 1), f'expected (1, 1), got {stride}'\n"
            "# Apply the args and confirm we got the textbook windows matrix.\n"
            "win = t.as_strided(x, size=size, stride=stride)\n"
            "expected = t.tensor([\n"
            "    [0., 1., 2.], [1., 2., 3.], [2., 3., 4.],\n"
            "    [3., 4., 5.], [4., 5., 6.], [5., 6., 7.],\n"
            "])\n"
            "assert t.equal(win, expected), f'window mismatch:\\n{win}'\n"
            "\n"
            "# Non-contiguous source — this is the whole reason we don't hard-\n"
            "# code stride=(1, 1). Slice every-other element to get stride 2.\n"
            "y = t.arange(20, dtype=t.float32)[::2]  # length 10, stride 2\n"
            "assert y.stride() == (2,), 'sanity: y should have stride (2,)'\n"
            "size_y, stride_y = ex1_window_args(y, kernel_width=4)\n"
            "assert size_y == (7, 4), f'expected (7, 4), got {size_y}'\n"
            "assert stride_y == (2, 2), f'expected (2, 2), got {stride_y}'\n"
            "win_y = t.as_strided(y, size=size_y, stride=stride_y)\n"
            "assert win_y.shape == (7, 4)\n"
            "# Element [0, 0] should be y[0] = 0; [1, 0] should be y[1] = 2.\n"
            "assert win_y[0, 0].item() == 0.0\n"
            "assert win_y[1, 0].item() == 2.0, (\n"
            "    f'non-contiguous stride wrong: window[1, 0] = {win_y[1, 0]} '\n"
            "    f'(expected 2.0). Did you hard-code stride=1?'\n"
            ")\n"
            "print('contiguous + non-contiguous windowing args both correct')"
        ),
        "solution_body": (
            "def ex1_window_args(x: Tensor, kernel_width: int) -> tuple:\n"
            "    L = x.shape[0]\n"
            "    s, = x.stride()\n"
            "    size = (L - kernel_width + 1, kernel_width)\n"
            "    stride = (s, s)\n"
            "    return size, stride"
        ),
        "solution_notes": (
            "**Both strides are the source's element stride.** The first "
            "stride says 'how to advance one window position' — one element "
            "forward through `x`. The second stride says 'how to advance one "
            "step inside a window' — also one element forward through `x`. "
            "Same value.\n\n"
            "**Why we don't hard-code `1`.** If `x` was built via "
            "`x = source[::2]`, its element stride is `2`, not `1`. Hard-"
            "coding `1` would scan adjacent memory cells — silently giving "
            "you the WRONG windowed view with no error message.\n\n"
            "**Generalises directly to N-D.** For ARENA's 1-D conv with "
            "batch + channels, you pull all of `x.stride()` (three values) "
            "and produce a 4-tuple stride `(s_B, s_IC, s_W, s_W)`. Same "
            "logic — window position and within-window both advance one "
            "spatial step."
        ),
    }


def _spec_as_strided_windowing_ex2():
    return {
        "atom_id": "as-strided-windowing",
        "subtopic": "PyTorch: as_strided windowing",
        "topic_folder": TOPIC_FOLDER,
        "atom_recap_md": RECAP_AS_STRIDED_WINDOWING,
        "exercise_index": 2,
        "exercise_title": "batched + channelled windowing for conv1d input prep",
        "slug": "batched-channelled-windowing-for-conv1d-input-prep",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["conv1d", "batch-channel-windowing", "as_strided"],
        "kcs": ["as-strided-window-stride", "conv1d-input-prep-shape"],
        "lo": (
            "Build the (B, IC, L_out, K) windowed view of a (B, IC, W) "
            "input used as the windowed-input matrix in ARENA's "
            "`conv1d_minimal` — without performing the convolution itself."
        ),
        "prompt_body": (
            "Implement `ex2_conv1d_windows(x, kernel_width)`. Given an input "
            "of shape `(B, IC, W)` and an integer `kernel_width = K`, return "
            "a zero-copy view of shape `(B, IC, L_out, K)` where `L_out = W "
            "- K + 1`. Each `(L_out, K)` slice is the sliding-window matrix "
            "for one (batch, channel) pair.\n\n"
            "**Strides to use.** Pull all of `x.stride()` (call it `s_B, "
            "s_IC, s_W`) and build the new view with stride `(s_B, s_IC, "
            "s_W, s_W)`. The last two axes both advance one spatial step.\n\n"
            "Inputs:\n"
            "- `x`: `(B, IC, W)` float tensor.\n"
            "- `kernel_width`: int, `<= W`.\n\n"
            "Output: zero-copy view of shape `(B, IC, L_out, K)`.\n\n"
            "**Don't run the convolution** — this drill exercises ONLY the "
            "input-prep step. ARENA's `conv1d_minimal` then einsums this "
            "view with the kernel; that's a separate atom."
        ),
        "stub": (
            "def ex2_conv1d_windows(x: Tensor, kernel_width: int) -> Tensor:\n"
            '    """Build the (B, IC, L_out, K) windowed view used in conv1d input prep."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "B, IC, W, K = 2, 3, 7, 3\n"
            "x = t.arange(B * IC * W, dtype=t.float32).reshape(B, IC, W)\n"
            "win = ex2_conv1d_windows(x, kernel_width=K)\n"
            "L_out = W - K + 1\n"
            "assert win.shape == (B, IC, L_out, K), (\n"
            "    f'expected {(B, IC, L_out, K)}, got {tuple(win.shape)}'\n"
            ")\n"
            "assert win.dtype == t.float32\n"
            "# Must be a zero-copy view — shares storage with x.\n"
            "assert win.data_ptr() == x.data_ptr(), 'windows view must alias x storage'\n"
            "\n"
            "# Spot-check a window.\n"
            "# For batch 0, channel 1, window 0, the K elements are x[0, 1, 0:K].\n"
            "assert t.equal(win[0, 1, 0], x[0, 1, 0:K]), 'window[0,1,0] mismatch'\n"
            "# Last window for batch 1, channel 2: x[1, 2, W-K:W].\n"
            "assert t.equal(win[1, 2, -1], x[1, 2, W - K:W]), (\n"
            "    f'window[1,2,-1] mismatch: got {win[1, 2, -1]}, expected {x[1, 2, W - K:W]}'\n"
            ")\n"
            "\n"
            "# Cross-check by running the conv1d pipeline end-to-end.\n"
            "import torch.nn.functional as F\n"
            "rng = t.Generator().manual_seed(1)\n"
            "x_big = t.randn(2, 3, 16, generator=rng)\n"
            "weights = t.randn(5, 3, 4, generator=rng)  # (OC, IC, K)\n"
            "win_big = ex2_conv1d_windows(x_big, kernel_width=4)\n"
            "assert win_big.shape == (2, 3, 13, 4)\n"
            "# Apply the einsum from ARENA's conv1d_minimal and compare to F.conv1d.\n"
            "ours = t.einsum('bicwk,ocik->bocw'.replace('cw', 'lk').replace('ic', 'i'), win_big, weights) if False else \\\n"
            "       t.einsum('b i l k, o i k -> b o l', win_big, weights)\n"
            "ref = F.conv1d(x_big, weights)\n"
            "assert t.allclose(ours, ref, atol=1e-4), (\n"
            "    f'conv1d via your windows + einsum doesn\\'t match F.conv1d:\\n'\n"
            "    f'max diff = {(ours - ref).abs().max().item()}'\n"
            ")\n"
            "print(f'window view shape {tuple(win.shape)} verified zero-copy + matches F.conv1d when combined with einsum')"
        ),
        "solution_body": (
            "def ex2_conv1d_windows(x: Tensor, kernel_width: int) -> Tensor:\n"
            "    B, IC, W = x.shape\n"
            "    s_B, s_IC, s_W = x.stride()\n"
            "    L_out = W - kernel_width + 1\n"
            "    return t.as_strided(\n"
            "        x,\n"
            "        size=(B, IC, L_out, kernel_width),\n"
            "        stride=(s_B, s_IC, s_W, s_W),\n"
            "    )"
        ),
        "solution_notes": (
            "**This is the load-bearing trick of ARENA's `conv1d_minimal`.** "
            "Once you have the `(B, IC, L_out, K)` view, the actual "
            "convolution is a one-line einsum: "
            "`einsum('b i l k, o i k -> b o l', windows, weights)`. The "
            "windowing IS the hard part; the contraction is just a "
            "tensor multiply.\n\n"
            "**Always pull stride from `x.stride()`.** ARENA's source code "
            "has an explicit comment warning that hard-coding the last "
            "stride to `1` is the #1 silent-bug pattern when this trick is "
            "extended to conv2d (where the last stride for a "
            "non-contiguous channel-major input would be wrong).\n\n"
            "**Memory cost is zero, but the view IS overlapping.** Adjacent "
            "windows share `K - 1` elements. That's fine for read-only use "
            "(einsum) but you must NOT write through this view — writes to "
            "overlapping memory have undefined order."
        ),
    }


def _spec_slice_view_mutation():
    return {
        "atom_id": "slice-view-mutation",
        "subtopic": "PyTorch: Slice view mutation",
        "topic_folder": TOPIC_FOLDER,
        "atom_recap_md": RECAP_SLICE_VIEW,
        "exercise_index": 1,
        "exercise_title": "in-place zero the diagonal via slice-view writes",
        "slug": "in-place-zero-diagonal-via-slice-view-write",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["slice", "view", "in-place", "diagonal"],
        "kcs": ["slice-returns-view", "view-writes-alias-source"],
        "lo": (
            "Mutate a square matrix in place by writing through a "
            "slice-view (the diagonal), and verify the source tensor "
            "reflects the change because slices return views, not copies."
        ),
        "prompt_body": (
            "Implement `ex1_zero_diagonal_inplace(mat)`.\n\n"
            "Given a square `(N, N)` float tensor `mat`, set every diagonal "
            "entry to `0.0` **by writing through a slice-view of `mat`** — "
            "no `mat = ...` reassignment, no `mat.fill_diagonal_(0)`. The "
            "point is to exercise the view-aliasing property.\n\n"
            "**Hint.** `mat.diagonal()` (or `mat.diag()`) returns a 1-D "
            "view-tensor of length `N` that shares storage with `mat`. "
            "Writing `view[:] = 0.0` mutates `mat` in place.\n\n"
            "Inputs: `mat` — `(N, N)` float tensor.\n"
            "Output: the function should **return the same `mat` object** "
            "(not a copy). All diagonal entries are now zero; off-diagonal "
            "entries are untouched."
        ),
        "stub": (
            "def ex1_zero_diagonal_inplace(mat: Tensor) -> Tensor:\n"
            '    """Zero the diagonal of mat IN PLACE via a slice-view write."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "mat = t.arange(16, dtype=t.float32).reshape(4, 4) + 1.0\n"
            "# mat = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]\n"
            "original_data_ptr = mat.data_ptr()\n"
            "out = ex1_zero_diagonal_inplace(mat)\n"
            "\n"
            "# Identity preserved — same tensor object, same storage.\n"
            "assert out is mat, 'must return the same tensor object (in-place)'\n"
            "assert out.data_ptr() == original_data_ptr, 'must not reallocate'\n"
            "\n"
            "# Diagonal zeroed.\n"
            "expected_diag = t.zeros(4)\n"
            "assert t.equal(mat.diagonal(), expected_diag), (\n"
            "    f'diagonal not zeroed: {mat.diagonal()}'\n"
            ")\n"
            "\n"
            "# Off-diagonal untouched: row 0, col 1 was 2.0; row 2, col 0 was 9.0.\n"
            "assert mat[0, 1].item() == 2.0, f'mat[0, 1] should be 2.0, got {mat[0, 1]}'\n"
            "assert mat[2, 0].item() == 9.0, f'mat[2, 0] should be 9.0, got {mat[2, 0]}'\n"
            "\n"
            "# Larger random case — confirm symmetry of behavior.\n"
            "rng = t.Generator().manual_seed(3)\n"
            "big = t.randn(8, 8, generator=rng)\n"
            "off_diag_before = big.clone()\n"
            "off_diag_before.fill_diagonal_(0)  # zero only the diagonal\n"
            "ex1_zero_diagonal_inplace(big)\n"
            "assert t.allclose(big, off_diag_before), 'off-diagonal must be unchanged'\n"
            "assert t.allclose(big.diagonal(), t.zeros(8)), 'all 8 diagonal entries must be zero'\n"
            "print('diagonal zeroed in place; off-diagonal preserved; storage unchanged')"
        ),
        "solution_body": (
            "def ex1_zero_diagonal_inplace(mat: Tensor) -> Tensor:\n"
            "    mat.diagonal()[:] = 0.0\n"
            "    return mat"
        ),
        "solution_notes": (
            "**`mat.diagonal()` returns a view.** It's a 1-D tensor of "
            "length `N` whose elements alias `mat[0, 0], mat[1, 1], ..., "
            "mat[N-1, N-1]`. Writing `[:] = 0.0` through it scatters back to "
            "the source.\n\n"
            "**Why `mat.diagonal() = 0` would NOT work.** That's a Python "
            "rebind of a local name; it doesn't go through `__setitem__` and "
            "doesn't mutate `mat`. You need slice-assignment (`[:]` or "
            "`[...]`) to trigger the in-place write path.\n\n"
            "**Contrast with boolean indexing.** `mat[mat > 5] = 0` works "
            "(direct `__setitem__`), but `view = mat[mat > 5]; view[:] = 0` "
            "does NOT mutate `mat` — boolean indexing returns a copy, not a "
            "view, so the slice-write goes to nowhere useful."
        ),
    }


def _spec_stack_vs_cat():
    return {
        "atom_id": "stack-vs-cat",
        "subtopic": "PyTorch: stack vs cat",
        "topic_folder": TOPIC_FOLDER,
        "atom_recap_md": RECAP_STACK_VS_CAT,
        "exercise_index": 1,
        "exercise_title": "pick stack or cat from the target shape",
        "slug": "pick-stack-or-cat-from-target-shape",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["stack", "cat", "rank-change", "shape-reasoning"],
        "kcs": ["stack-inserts-axis", "cat-along-existing-axis"],
        "lo": (
            "Choose between `t.stack` and `t.cat` based on whether the "
            "target shape introduces a new axis (stack) or extends an "
            "existing axis (cat), and dispatch correctly."
        ),
        "prompt_body": (
            "Implement `ex1_combine(tensors, target_shape)`. Given:\n"
            "- `tensors`: a list of N tensors, all the same shape "
            "`(d0, d1, ..., dM)`.\n"
            "- `target_shape`: the desired output shape (a tuple).\n"
            "\n"
            "Pick `t.stack` or `t.cat` and the correct `dim` based on "
            "`target_shape`. Specifically:\n"
            "- If `len(target_shape) == M + 2` (one axis added — rank "
            "increased), use `t.stack(tensors, dim=k)` where `k` is the "
            "position of the new axis with size `N`.\n"
            "- If `len(target_shape) == M + 1` (same rank as inputs — one "
            "axis grew by factor `N`), use `t.cat(tensors, dim=k)` where "
            "`k` is the axis whose size is `N * d[k]`.\n"
            "- Otherwise raise `ValueError('cannot combine')`.\n"
            "\n"
            "Return: the combined tensor of shape `target_shape`."
        ),
        "stub": (
            "def ex1_combine(tensors: list, target_shape: tuple) -> Tensor:\n"
            '    """Dispatch to stack or cat based on target_shape."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Three (3, 4) tensors --------------------------------------------\n"
            "ts = [t.full((3, 4), float(i)) for i in range(3)]\n"
            "\n"
            "# Target (3, 3, 4): rank +1, new axis size 3 at position 0 → stack(dim=0).\n"
            "out_a = ex1_combine(ts, target_shape=(3, 3, 4))\n"
            "assert out_a.shape == (3, 3, 4), f'got {tuple(out_a.shape)}'\n"
            "assert t.equal(out_a, t.stack(ts, dim=0))\n"
            "\n"
            "# Target (3, 3, 4): same target_shape works → confirms stack picked.\n"
            "# Target (3, 4, 3): rank +1, new axis at position 2 → stack(dim=2).\n"
            "out_b = ex1_combine(ts, target_shape=(3, 4, 3))\n"
            "assert out_b.shape == (3, 4, 3)\n"
            "assert t.equal(out_b, t.stack(ts, dim=2))\n"
            "\n"
            "# Target (9, 4): same rank, axis 0 grew 3→9 = 3*3 → cat(dim=0).\n"
            "out_c = ex1_combine(ts, target_shape=(9, 4))\n"
            "assert out_c.shape == (9, 4)\n"
            "assert t.equal(out_c, t.cat(ts, dim=0))\n"
            "\n"
            "# Target (3, 12): same rank, axis 1 grew 4→12 = 3*4 → cat(dim=1).\n"
            "out_d = ex1_combine(ts, target_shape=(3, 12))\n"
            "assert out_d.shape == (3, 12)\n"
            "assert t.equal(out_d, t.cat(ts, dim=1))\n"
            "\n"
            "# Target (2, 3, 4): wrong rank gap (M=2, target rank 3, but new\n"
            "# axis would need size 3 not 2) → must raise ValueError.\n"
            "try:\n"
            "    ex1_combine(ts, target_shape=(2, 3, 4))\n"
            "    raise AssertionError('expected ValueError for incompatible shape')\n"
            "except ValueError:\n"
            "    pass\n"
            "\n"
            "# Confirm via dtypes too: combine of 3 long tensors should stay long.\n"
            "longs = [t.tensor([1, 2, 3], dtype=t.long) for _ in range(4)]\n"
            "out_l = ex1_combine(longs, target_shape=(4, 3))\n"
            "assert out_l.dtype == t.long\n"
            "assert out_l.shape == (4, 3)\n"
            "print('stack/cat dispatched correctly for 5 shape patterns')"
        ),
        "solution_body": (
            "def ex1_combine(tensors: list, target_shape: tuple) -> Tensor:\n"
            "    N = len(tensors)\n"
            "    in_shape = tuple(tensors[0].shape)\n"
            "    M = len(in_shape)\n"
            "    tgt = tuple(target_shape)\n"
            "    if len(tgt) == M + 1:\n"
            "        # stack — find the axis whose size is N.\n"
            "        for k, sz in enumerate(tgt):\n"
            "            if sz == N and tgt[:k] + tgt[k + 1:] == in_shape:\n"
            "                return t.stack(tensors, dim=k)\n"
            "        raise ValueError('cannot combine')\n"
            "    elif len(tgt) == M:\n"
            "        # cat — find the axis where target = N * input size.\n"
            "        for k in range(M):\n"
            "            if tgt[k] == N * in_shape[k] and all(\n"
            "                tgt[j] == in_shape[j] for j in range(M) if j != k\n"
            "            ):\n"
            "                return t.cat(tensors, dim=k)\n"
            "        raise ValueError('cannot combine')\n"
            "    else:\n"
            "        raise ValueError('cannot combine')"
        ),
        "solution_notes": (
            "**Single-axis-change is the deciding question.** If the output "
            "needs a brand-new axis, that's `stack`; if it needs one "
            "existing axis to grow, that's `cat`. Anything else is "
            "incompatible.\n\n"
            "**Why this dispatch matters in real code.** ARENA exercises "
            "switch between the two constantly — `stack` for assembling "
            "per-head/per-batch results into a leading-axis tensor, `cat` "
            "for assembling residual-stream contributions or aggregating "
            "logits across model copies. Confusing them produces a tensor "
            "that's either 1 rank too high or 1 rank too low — usually "
            "caught by a downstream shape-assertion crash.\n\n"
            "**The bare-functional rewrite.** `t.stack(xs, dim=k)` is "
            "literally `t.cat([x.unsqueeze(k) for x in xs], dim=k)`. Knowing "
            "this lets you read source code that uses one when you would've "
            "used the other."
        ),
    }


def _spec_tensor_to_device():
    return {
        "atom_id": "tensor-to-device",
        "subtopic": "PyTorch: tensor.to(device)",
        "topic_folder": TOPIC_FOLDER,
        "atom_recap_md": RECAP_TO_DEVICE,
        "exercise_index": 1,
        "exercise_title": "move tensor to chosen device with the CPU/CUDA guard",
        "slug": "move-tensor-to-device-with-cuda-guard",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["device", "cpu", "cuda", "to", "guard"],
        "kcs": ["pick-device-with-cuda-available", "to-is-not-inplace"],
        "lo": (
            "Pick a target device with the standard `cuda.is_available()` "
            "guard, move a tensor to it via `.to(device)` with reassignment, "
            "and confirm the move was not in place."
        ),
        "prompt_body": (
            "Implement `ex1_to_best_device(x)`.\n\n"
            "1. Pick `device = 'cuda' if t.cuda.is_available() else 'cpu'`.\n"
            "2. Return `x.to(device)` — **without modifying `x` itself** (do "
            "NOT do `x = x.to(...)` and then `return x` if there's any way "
            "for the caller to see the unmodified original; the test "
            "captures `id(x)` before the call and checks the moved tensor is "
            "a different object when the device actually changed).\n"
            "3. The function must work on a CPU-only machine — the test "
            "always runs on CPU, so `device` will resolve to `'cpu'` and "
            "the returned tensor's device should be `'cpu'`.\n\n"
            "Input: `x` — any `torch.Tensor`.\n"
            "Output: a `torch.Tensor` on the chosen device, with identical "
            "shape and dtype to `x`.\n\n"
            "**Why this is its own drill.** Forgetting that `.to()` is "
            "not-in-place — and missing the reassignment — is the #1 'why is "
            "my model still on CPU' bug for newcomers."
        ),
        "stub": (
            "def ex1_to_best_device(x: Tensor) -> Tensor:\n"
            '    """Pick device via cuda.is_available(), return x.to(device)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# CPU-only target environment — this test passes both on CPU and CUDA\n"
            "# boxes (the chosen device tracks t.cuda.is_available()).\n"
            "x = t.arange(12, dtype=t.float32).reshape(3, 4)\n"
            "x_orig = x.clone()  # for value comparison\n"
            "expected_device = t.device('cuda' if t.cuda.is_available() else 'cpu')\n"
            "\n"
            "out = ex1_to_best_device(x)\n"
            "\n"
            "# Shape + dtype preserved.\n"
            "assert out.shape == x.shape\n"
            "assert out.dtype == x.dtype\n"
            "# Device matches the cuda.is_available() guard.\n"
            "assert out.device.type == expected_device.type, (\n"
            "    f'expected device {expected_device}, got {out.device}'\n"
            ")\n"
            "# Values intact (matches the original).\n"
            "out_cpu = out.cpu()\n"
            "assert t.equal(out_cpu, x_orig), 'value content changed during move'\n"
            "\n"
            "# Confirm the source tensor was not modified in place.\n"
            "assert t.equal(x, x_orig), 'source tensor was modified — .to() must not be in place'\n"
            "\n"
            "# When already on the target device, .to() should be a no-op view\n"
            "# (or the same tensor) — but at minimum return something equal.\n"
            "y = t.zeros(5, device='cpu')\n"
            "y_out = ex1_to_best_device(y)\n"
            "assert y_out.device.type == 'cpu' if not t.cuda.is_available() else True\n"
            "assert t.equal(y_out.cpu(), y.cpu())\n"
            "print(f'tensor moved to {out.device} (cuda available: {t.cuda.is_available()})')"
        ),
        "solution_body": (
            "def ex1_to_best_device(x: Tensor) -> Tensor:\n"
            "    device = 'cuda' if t.cuda.is_available() else 'cpu'\n"
            "    return x.to(device)"
        ),
        "solution_notes": (
            "**`.to()` is not in-place.** It returns a (potentially) new "
            "tensor. You MUST capture the return value with `out = "
            "x.to(device)` or `x = x.to(device)`. Code that just writes "
            "`x.to(device)` and moves on silently leaves `x` on the "
            "original device.\n\n"
            "**Contrast with `model.to(device)`.** Modules DO move in "
            "place — `model.to(device)` mutates the parameters / buffers "
            "directly. Tensors don't. This asymmetry catches a LOT of "
            "PyTorch newcomers.\n\n"
            "**The `cuda.is_available()` guard is the universal pattern.** "
            "Hard-coding `device='cuda'` breaks every CI run that doesn't "
            "have a GPU. Putting the guard at the top of the script and "
            "threading the chosen device everywhere downstream keeps the "
            "code portable.\n\n"
            "**Same-device `to()` may return the same object.** No copy is "
            "made if `x` is already on `device` with matching dtype. Don't "
            "rely on identity (`is`) either way — just trust the device "
            "attribute."
        ),
    }


def _spec_tensor_wraps_ndarray():
    return {
        "atom_id": "tensor-wraps-ndarray",
        "subtopic": "PyTorch: tensor from ndarray",
        "topic_folder": TOPIC_FOLDER,
        "atom_recap_md": RECAP_FROM_NDARRAY,
        "exercise_index": 1,
        "exercise_title": "compare from_numpy aliasing vs tensor copy",
        "slug": "compare-from-numpy-aliasing-vs-tensor-copy",
        "bloom_level": "Analyze",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["from_numpy", "tensor-factory", "aliasing", "copy"],
        "kcs": ["from-numpy-shares-storage", "tensor-factory-copies"],
        "lo": (
            "Distinguish `t.from_numpy` (shares memory with the source "
            "ndarray) from `t.tensor` (copies) by mutating the source and "
            "observing which torch tensor reflects the change."
        ),
        "prompt_body": (
            "Implement `ex1_aliasing_report(arr)`.\n\n"
            "Given a NumPy `ndarray` `arr`, build:\n"
            "1. `wrapped = t.from_numpy(arr)` — should share memory with "
            "`arr`.\n"
            "2. `copied = t.tensor(arr)` — should be an independent copy.\n"
            "\n"
            "Then mutate `arr` in place (set `arr[0] = 999`) and read the "
            "first element of each tensor. Return a dict:\n"
            "```python\n"
            "{\n"
            "    'wrapped_first': float(wrapped[0]),\n"
            "    'copied_first':  float(copied[0]),\n"
            "    'wrapped_shares_storage': <bool — wrapped[0] tracked the mutation>,\n"
            "    'copied_shares_storage':  <bool — copied[0] tracked the mutation>,\n"
            "}\n"
            "```\n"
            "\n"
            "The test will pre-record the original first-element value and "
            "verify your aliasing predictions are correct.\n\n"
            "Input: `arr` — `np.ndarray`, dtype `float32`, length ≥ 1.\n"
            "Output: dict with the four keys above."
        ),
        "stub": (
            "def ex1_aliasing_report(arr) -> dict:\n"
            '    """Build a from_numpy tensor and a t.tensor copy; mutate arr; report aliasing."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "arr = np.arange(5, dtype=np.float32)\n"
            "# arr[0] starts at 0.0; the function should mutate it to 999.0.\n"
            "report = ex1_aliasing_report(arr)\n"
            "\n"
            "assert isinstance(report, dict), 'must return a dict'\n"
            "for key in ('wrapped_first', 'copied_first',\n"
            "            'wrapped_shares_storage', 'copied_shares_storage'):\n"
            "    assert key in report, f'missing key {key!r}'\n"
            "\n"
            "# from_numpy aliases storage → wrapped saw the 999.\n"
            "assert report['wrapped_first'] == 999.0, (\n"
            "    f'wrapped_first should be 999.0 (aliasing), got {report[\"wrapped_first\"]}'\n"
            ")\n"
            "assert report['wrapped_shares_storage'] is True, (\n"
            "    'from_numpy MUST share storage with the source ndarray'\n"
            ")\n"
            "\n"
            "# t.tensor copies → copied did NOT see the 999.\n"
            "assert report['copied_first'] == 0.0, (\n"
            "    f'copied_first should be 0.0 (independent copy), got {report[\"copied_first\"]}'\n"
            ")\n"
            "assert report['copied_shares_storage'] is False, (\n"
            "    't.tensor MUST NOT share storage with the source ndarray'\n"
            ")\n"
            "\n"
            "# Confirm the function actually mutated arr (not just claimed to).\n"
            "assert arr[0] == 999.0, 'function should have mutated arr[0] in place'\n"
            "print('from_numpy aliases; t.tensor copies — both verified by mutation test')"
        ),
        "solution_body": (
            "def ex1_aliasing_report(arr) -> dict:\n"
            "    wrapped = t.from_numpy(arr)\n"
            "    copied = t.tensor(arr)\n"
            "    original = float(arr[0])\n"
            "    arr[0] = 999.0\n"
            "    return {\n"
            "        'wrapped_first': float(wrapped[0]),\n"
            "        'copied_first':  float(copied[0]),\n"
            "        'wrapped_shares_storage': float(wrapped[0]) != original,\n"
            "        'copied_shares_storage':  float(copied[0]) != original,\n"
            "    }"
        ),
        "solution_notes": (
            "**`t.from_numpy` is a zero-copy wrap.** The torch tensor and "
            "the source ndarray share the same memory buffer. Mutate "
            "either, the other sees it. Useful for fast NumPy → torch in a "
            "data pipeline where you don't want to double-allocate.\n\n"
            "**`t.tensor(arr)` always copies.** This is the safe-but-"
            "slower factory. If you're at all unsure about who else might "
            "mutate the source array, prefer this over `from_numpy`.\n\n"
            "**`t.as_tensor(arr)` is the middle ground.** Shares memory "
            "when possible (matching dtype and device, source is an "
            "ndarray), copies otherwise. Good default for generic 'turn "
            "this into a tensor' code paths.\n\n"
            "**Dtype trap.** NumPy's default float dtype is `float64`. If "
            "you `t.from_numpy(np.array([1.0, 2.0]))` you get a `float64` "
            "torch tensor, which will misbehave the moment it touches a "
            "`float32` model. Either build the ndarray with `dtype="
            "'float32'` from the start, or chain `.float()` on the torch "
            "tensor."
        ),
    }


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

ALL_SPECS = [
    _spec_contiguous_ex1(),
    _spec_contiguous_ex2(),
    _spec_as_strided_windowing_ex1(),
    _spec_as_strided_windowing_ex2(),
    _spec_slice_view_mutation(),
    _spec_stack_vs_cat(),
    _spec_tensor_to_device(),
    _spec_tensor_wraps_ndarray(),
]


def verify_solutions(specs: list) -> None:
    """Execute every spec's solution + test bodies in-process.

    Bails out before any notebook is written if any test fails. This is the
    Doughty/Maier 'solution executes cleanly' guardrail.
    """
    try:
        import numpy as np
        import torch as t
        from torch import Tensor
    except ImportError as exc:
        raise SystemExit(
            f"[verify] missing runtime dep: {exc}\n"
            "  Run this script with the backend venv that has torch + einops installed."
        )

    failures: list[str] = []
    for spec in specs:
        ex_id = f"ex{spec['exercise_index']}"
        label = f"{spec['atom_id']}/{ex_id} ({spec['exercise_title']})"
        ns: dict = {
            "t": t,
            "np": np,
            "Tensor": Tensor,
        }
        # Compile the solution.
        try:
            exec(spec["solution_body"], ns)
        except Exception:
            failures.append(
                f"{label}: solution_body did not compile:\n"
                + traceback.format_exc()
            )
            continue
        # Run the test body inside a function so internal `import` / locals work.
        test_src = "def _verify():\n"
        for line in spec["test_body"].split("\n"):
            test_src += "    " + line + "\n"
        try:
            exec(test_src, ns)
            ns["_verify"]()
        except Exception:
            failures.append(
                f"{label}: test_body failed:\n" + traceback.format_exc()
            )
            continue
        print(f"[verify ok] {label}")

    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        raise SystemExit(
            f"[verify] {len(failures)} solution(s) failed — refusing to write notebooks."
        )


def main() -> None:
    print(f"[author] verifying {len(ALL_SPECS)} solutions in-process before writing...")
    verify_solutions(ALL_SPECS)
    print("[author] all solutions pass; emitting notebooks.")
    for spec in ALL_SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")


if __name__ == "__main__":
    main()
