#!/usr/bin/env python3
"""Author Colab-native standalones for numerical-stability + module-pattern atoms.

Batch 3: covers 8 single-exercise atoms that ARENA chap-0 silently composes
across CNN / BatchNorm / LayerNorm / training-loop exercises:
    - stride-zero-broadcast            (1 ex) — PyTorch: Zero-stride broadcasting
    - sqrt-eps-stabilize               (1 ex) — Numerical: sqrt-eps stabilization
    - kaiming-uniform-init             (1 ex) — Init: Kaiming uniform
    - device-consistent-construct      (1 ex) — PyTorch: Device-consistent tensor construction
    - conditional-hparam-branch        (1 ex) — PyTorch: Conditional hparam branch
    - rearrange-as-sequential-layer    (1 ex) — Einops: Rearrange as nn.Sequential layer
    - encoder-decoder-symmetric        (1 ex) — CNN: Encoder-decoder symmetric layout
    - loss-item-scalar-extract         (1 ex) — PyTorch: loss.item() scalar extract

Brand-new folder `prereqs_numerical_modules/`. Each atom is the smallest
ARENA-composing skill that flashcards alone can't deliver (needs interactive
tensor execution, init-distribution viz, or device-aware sanity checks).

Inherits build-time verify gate — every canonical solution is exec'd against
its in-notebook test before any .ipynb is written. Aborts on first failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_numerical_modules"


# ===================================================================== recaps

RECAP_STRIDE_ZERO = (
    "## Zero-stride broadcasting — quick refresher\n"
    "\n"
    "When two tensors with mismatched shapes are combined elementwise, PyTorch "
    "broadcasts the smaller along the missing / size-1 axes. The implementation "
    "is dramatically cheaper than the user-facing semantics suggest: the smaller "
    "tensor's `.stride()` along the broadcast axis is set to **zero**, so the "
    "underlying memory is reused for every position along that axis — no copy, "
    "no allocation.\n"
    "\n"
    "```python\n"
    "x = torch.tensor([1., 2., 3.])           # shape (3,), strides (1,)\n"
    "y = x.expand(4, 3)                       # shape (4, 3), strides (0, 1)\n"
    "y.data_ptr() == x.data_ptr()             # True — same storage\n"
    "```\n"
    "\n"
    "**Three families to keep separate.**\n"
    "1. **Implicit broadcast** — happens during arithmetic (`a + b`, `a * b`). "
    "No new tensor materialized.\n"
    "2. **`.expand(shape)`** — creates a *view* with stride 0 on broadcast axes. "
    "Read-only in practice (writes alias all positions and usually error).\n"
    "3. **`.repeat(...)`** / `einops.repeat` — actually *copies* data so every "
    "position has its own memory. Strides are all nonzero. Use this when you "
    "need to write into the broadcasted result.\n"
    "\n"
    "**Why this matters.** A `(B, N, N)` attention mask broadcast from a `(N, N)` "
    "tensor adds zero memory cost. A `.repeat(B, 1, 1)` would multiply the mask "
    "footprint by `B` — a 4096-token transformer with batch 32 would balloon "
    "from 64 MB to 2 GB."
)

RECAP_SQRT_EPS = (
    "## `sqrt(x + eps)` stabilization — quick refresher\n"
    "\n"
    "Every BatchNorm / LayerNorm / RMSNorm / Adam / `pairwise_distance` "
    "implementation contains the same one-character defense:\n"
    "\n"
    "```python\n"
    "normed = x / torch.sqrt(var + eps)        # NOT torch.sqrt(var) — would divide by ~0\n"
    "```\n"
    "\n"
    "Without `+ eps`, three things can go wrong when `var` (or any other "
    "non-negative quantity) collapses toward zero:\n"
    "1. `sqrt(0) == 0` → divide-by-zero → `inf` propagates everywhere.\n"
    "2. `sqrt(tiny_positive)` → numerically unstable gradient — `d/dx sqrt(x) = "
    "1 / (2*sqrt(x))` blows up near 0.\n"
    "3. `sqrt(tiny_negative_from_float_roundoff)` → `nan` (PyTorch returns NaN "
    "on negative inputs, not an exception).\n"
    "\n"
    "**Where to put the eps.** *Inside* the sqrt (`sqrt(var + eps)`), not after "
    "(`sqrt(var) + eps`). The inside-sqrt form keeps the derivative bounded; "
    "the outside form does not — it only protects against the divide-by-zero, "
    "not against the gradient explosion.\n"
    "\n"
    "**Typical eps.** `1e-5` for BatchNorm, `1e-6` for LayerNorm/RMSNorm, "
    "`1e-8` for Adam's denominator. The values come from empirical stability "
    "on float32; pick smaller eps only if you've verified the input range."
)

RECAP_KAIMING = (
    "## Kaiming-uniform init — quick refresher\n"
    "\n"
    "PyTorch's default `nn.Linear` / `nn.Conv2d` init is Kaiming-uniform with "
    "`a=sqrt(5)`, which simplifies to:\n"
    "\n"
    "```python\n"
    "bound = 1 / math.sqrt(fan_in)\n"
    "weight.uniform_(-bound, +bound)\n"
    "```\n"
    "\n"
    "where `fan_in = in_features` for Linear, `in_channels * kH * kW` for "
    "Conv2d. The general He/Kaiming formula `Uniform(-sqrt(6 / fan_in), "
    "+sqrt(6 / fan_in))` corresponds to `a=0` (pure ReLU) — PyTorch uses the "
    "`a=sqrt(5)` variant for historical compatibility, which gives the cleaner "
    "`1/sqrt(fan_in)` bound used above.\n"
    "\n"
    "**Why uniform, not normal.** Both give variance `~2 / fan_in` for the "
    "ReLU-flavored variant, but uniform has bounded support — no extreme "
    "outliers at init, which empirically helps very deep networks converge.\n"
    "\n"
    "**Why `fan_in` and not `fan_out`.** Forward-pass variance preservation. "
    "If you also care about backward-pass variance, average them (`fan_avg`). "
    "PyTorch's default uses `fan_in` because for most architectures forward "
    "stability matters more than backward.\n"
    "\n"
    "**The bias.** PyTorch initializes bias from the same `Uniform(-bound, "
    "+bound)` distribution as the weight (using the weight's fan_in). "
    "`nn.init.kaiming_uniform_` only touches the weight; you handle bias "
    "yourself with a matching `.uniform_(-bound, +bound)`."
)

RECAP_DEVICE_CONSISTENT = (
    "## Device-consistent tensor construction — quick refresher\n"
    "\n"
    "Custom Modules often allocate auxiliary tensors inside `forward` "
    "(temporary masks, running stats, scratch buffers). Two patterns differ "
    "subtly in performance:\n"
    "\n"
    "```python\n"
    "# Pattern A (BAD) — allocates on CPU then transfers\n"
    "buf = torch.zeros(x.shape).to(x.device)\n"
    "\n"
    "# Pattern B (GOOD) — allocates directly on the right device + dtype\n"
    "buf = torch.zeros(x.shape, device=x.device, dtype=x.dtype)\n"
    "```\n"
    "\n"
    "Pattern A does a CPU malloc + zero-fill + host→device copy on every "
    "forward pass. Pattern B does one device-side `cudaMalloc` (or equivalent) "
    "and skips the copy. On a tight inner loop this gap is hundreds of "
    "microseconds → measurable training-step speedup.\n"
    "\n"
    "**The dtype half of the rule matters too.** If `x` is `float16` (mixed-"
    "precision training) and you allocate `torch.zeros(shape)` (defaults to "
    "`float32`), the subsequent arithmetic upcasts — wasting half the memory "
    "savings amp was supposed to deliver. Always thread `dtype=x.dtype` "
    "through.\n"
    "\n"
    "**Sibling helpers.** `torch.zeros_like(x)`, `torch.ones_like(x)`, "
    "`torch.empty_like(x)`, `torch.full_like(x, val)` do this in one call — "
    "they inherit device + dtype + memory-layout from `x`. Use these when the "
    "new tensor's shape matches `x`."
)

RECAP_COND_HPARAM = (
    "## Conditional hparam branch — quick refresher\n"
    "\n"
    "Many `nn.Module` configs have an optional component — bias, dropout, "
    "layer-norm — gated by a boolean hyperparameter. The canonical pattern "
    "uses `if` inside `__init__` to either register the sub-component as a "
    "Parameter/Module or set the slot to `None`:\n"
    "\n"
    "```python\n"
    "class Block(nn.Module):\n"
    "    def __init__(self, dim, use_bias=True, dropout=0.0):\n"
    "        super().__init__()\n"
    "        self.linear = nn.Linear(dim, dim, bias=use_bias)\n"
    "        if dropout > 0:\n"
    "            self.dropout = nn.Dropout(dropout)\n"
    "        else:\n"
    "            self.dropout = None\n"
    "    def forward(self, x):\n"
    "        x = self.linear(x)\n"
    "        if self.dropout is not None:\n"
    "            x = self.dropout(x)\n"
    "        return x\n"
    "```\n"
    "\n"
    "**Why set to `None`, not just skip the assignment.** A consistent slot "
    "name (`self.dropout`) lets `forward` always reference it; the `is not "
    "None` guard then short-circuits cleanly. Skipping the assignment would "
    "raise `AttributeError` on the `forward` branch, which is the worst kind "
    "of runtime failure — bypasses every static-analysis tool.\n"
    "\n"
    "**`nn.Identity()` is the alternative.** `self.dropout = nn.Dropout(p) "
    "if p > 0 else nn.Identity()` gives you an always-callable slot — "
    "`forward` becomes `x = self.dropout(x)` unconditionally. Pick this when "
    "the conditional branch is hot (avoid the per-call `is not None` check) "
    "and the no-op sub-module's tiny overhead is fine."
)

RECAP_REARRANGE_LAYER = (
    "## `einops.layers.torch.Rearrange` as nn.Sequential layer — quick refresher\n"
    "\n"
    "Inside a `forward()` you'd write `einops.rearrange(x, 'b c h w -> b (c h "
    "w)')`. But when composing inside `nn.Sequential`, you need a Module, not "
    "a function. `einops.layers.torch.Rearrange` is the answer:\n"
    "\n"
    "```python\n"
    "from einops.layers.torch import Rearrange\n"
    "\n"
    "model = nn.Sequential(\n"
    "    nn.Conv2d(3, 32, 3, padding=1),\n"
    "    nn.ReLU(),\n"
    "    Rearrange('b c h w -> b (c h w)'),    # the 'Flatten' step\n"
    "    nn.Linear(32 * 28 * 28, 10),\n"
    ")\n"
    "```\n"
    "\n"
    "**Why this beats `nn.Flatten()`.** Flatten is opaque — you have to read "
    "the docs to remember its `start_dim` / `end_dim` semantics. `Rearrange` "
    "puts the shape transformation in the source as an algebraic string; the "
    "next reader sees `b c h w -> b (c h w)` and knows exactly what shape "
    "comes out.\n"
    "\n"
    "**There are three sibling Modules.** `Rearrange`, `Reduce`, and the rare "
    "`EinMix`. All live in `einops.layers.torch` (or `.tensorflow`, `.flax`). "
    "Use `Reduce` for global-average-pool / channel-mean / softmax-stabilize "
    "inside Sequential, same way.\n"
    "\n"
    "**Gotcha — not the same import path.** `from einops import rearrange` "
    "gets you the *function*. `from einops.layers.torch import Rearrange` "
    "(capital R, different module) gets you the *layer*. They share the same "
    "string grammar; only the wrapping differs."
)

RECAP_ENC_DEC = (
    "## Encoder-decoder symmetric layout — quick refresher\n"
    "\n"
    "Autoencoders, U-Nets, and segmentation heads all share the same "
    "structural pattern: every encoder downsample is mirrored by a decoder "
    "upsample, so the output shape == input shape.\n"
    "\n"
    "```\n"
    "encoder:    Conv → Conv → Pool  → Conv → Conv → Pool  → ...     (spatial /= 2 per stage)\n"
    "decoder:    ConvT → ConvT → Up  → ConvT → ConvT → Up  → ...     (spatial *= 2 per stage)\n"
    "```\n"
    "\n"
    "**Two upsampling Modules.** `nn.ConvTranspose2d` (learnable, can fix the "
    "checkerboard via odd kernels) and `nn.Upsample(scale_factor=2)` + a "
    "follow-up `Conv2d` (non-learnable upsample then learnable convolution — "
    "cleaner artifacts in practice). U-Net uses the Upsample+Conv variant.\n"
    "\n"
    "**Why symmetry matters beyond just shape.** Symmetric layouts let you "
    "**skip-connect** matching encoder/decoder stages — that's how U-Net "
    "preserves spatial detail through the bottleneck. If your encoder and "
    "decoder stage counts diverge, you can't lay down those skips.\n"
    "\n"
    "**Channel mirror.** Channel counts also mirror: encoder doubles channels "
    "per downsample (`3 → 16 → 32 → 64`), decoder halves them per upsample "
    "(`64 → 32 → 16 → 3`). Bottom-of-the-U layer keeps the highest channel "
    "count.\n"
    "\n"
    "**Shape parity test.** A correctly-laid-out encoder-decoder Module must "
    "satisfy `model(x).shape == x.shape` for any valid input. This single "
    "assertion catches almost every off-by-one in pool/stride/padding."
)

RECAP_LOSS_ITEM = (
    "## `loss.item()` scalar extraction — quick refresher\n"
    "\n"
    "Inside a training loop you frequently need the loss as a Python float — "
    "to print, to log, to compare against a threshold. **`.item()` is the only "
    "correct way** for 0-D (scalar) tensors:\n"
    "\n"
    "```python\n"
    "loss = F.cross_entropy(logits, labels)   # 0-D tensor on GPU\n"
    "scalar = loss.item()                     # Python float, autograd-detached\n"
    "wandb.log({'loss': scalar})\n"
    "```\n"
    "\n"
    "**Three patterns, three jobs.**\n"
    "1. **`.item()`** — extracts ONE scalar. Synchronizes (forces CUDA → CPU "
    "stall). Use for logging / branching. Errors on multi-element tensors.\n"
    "2. **`.detach().cpu()`** — keeps the tensor structure, just removes "
    "autograd + moves to CPU. Use when you want to **buffer** many step values "
    "(append to a list, then `torch.stack` later) without holding the compute "
    "graph hostage.\n"
    "3. **`float(loss)`** — works for 0-D tensors via the `__float__` dunder, "
    "but PyTorch deprecated this for >=0-D ambiguity. Use `.item()` instead.\n"
    "\n"
    "**Why `.item()` synchronizes.** It must wait for the kernel that "
    "produced the loss to finish before reading the value back. Calling "
    "`.item()` every step is fine; calling it on intermediate activations "
    "inside the inner loop tanks throughput.\n"
    "\n"
    "**The autograd half.** `.item()` automatically detaches — the Python "
    "float has no `.grad_fn`. This is what makes it safe to store: "
    "`loss_history.append(loss)` would pin the compute graph forever; "
    "`loss_history.append(loss.item())` is a one-time cost."
)


# ===================================================================== SPECS

SPECS = [
    # ============================================================ stride-zero-broadcast / ex1
    {
        "atom_id": "stride-zero-broadcast",
        "subtopic": "PyTorch: Zero-stride broadcasting",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_STRIDE_ZERO,
        "exercise_index": 1,
        "exercise_title": "diagnose zero-stride vs copy via .stride() + storage check",
        "slug": "diagnose-zero-stride-vs-copy",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["broadcast", "stride", "expand", "repeat", "memory"],
        "kcs": ["broadcast-stride-zero-readout", "expand-vs-repeat-memory"],
        "lo": (
            "Distinguish a zero-stride broadcast view from a true memory copy "
            "by reading `.stride()` and comparing `.data_ptr()` on both forms."
        ),
        "prompt_body": (
            "Implement `ex1_classify_broadcasts(x)` to characterize the two "
            "ways of replicating a 1-D vector into a 2-D matrix:\n\n"
            "1. `x` is a 1-D `(N,)` float tensor (e.g. `t.tensor([1., 2., 3.])`).\n"
            "2. Produce `expanded = x.expand(4, N)` — the *view* form.\n"
            "3. Produce `repeated = x.repeat(4, 1)` — the *copy* form.\n"
            "4. Return a dict with EXACTLY these keys:\n"
            "   - `'expanded'`: the expanded tensor.\n"
            "   - `'repeated'`: the repeated tensor.\n"
            "   - `'expanded_stride'`: `tuple(expanded.stride())` — must be "
            "`(0, 1)` for an N-vector broadcast to `(4, N)`.\n"
            "   - `'repeated_stride'`: `tuple(repeated.stride())` — must be "
            "`(N, 1)` (or whatever the contiguous (4, N) strides are).\n"
            "   - `'expanded_shares_storage'`: bool — `expanded.data_ptr() == "
            "x.data_ptr()`. Should be `True`.\n"
            "   - `'repeated_shares_storage'`: bool — `repeated.data_ptr() == "
            "x.data_ptr()`. Should be `False` (a copy was made).\n\n"
            "The point: both produce a `(4, N)` tensor with the same values, "
            "but `.expand()` allocates ZERO new memory (stride 0 on the new "
            "axis points back at the original storage), while `.repeat()` "
            "copies `4 * N` floats into fresh memory.\n\n"
            "**The print** at the end of your function should show the strides "
            "and data-ptr comparison so the caller sees the diagnostic."
        ),
        "stub": (
            "def ex1_classify_broadcasts(x: Tensor) -> dict:\n"
            '    """Return diagnostic dict comparing .expand() (view, stride 0) vs .repeat() (copy, nonzero stride)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x = t.tensor([1.0, 2.0, 3.0])\n"
            "out = ex1_classify_broadcasts(x)\n"
            "assert isinstance(out, dict), f'expected dict, got {type(out).__name__}'\n"
            "required_keys = {'expanded', 'repeated', 'expanded_stride', 'repeated_stride',\n"
            "                 'expanded_shares_storage', 'repeated_shares_storage'}\n"
            "assert set(out.keys()) == required_keys, f'expected keys {required_keys}, got {set(out.keys())}'\n"
            "\n"
            "# Both forms must give the same values + shape.\n"
            "assert out['expanded'].shape == (4, 3), f'expanded shape wrong: {tuple(out[\"expanded\"].shape)}'\n"
            "assert out['repeated'].shape == (4, 3), f'repeated shape wrong: {tuple(out[\"repeated\"].shape)}'\n"
            "expected_values = t.tensor([[1., 2., 3.]] * 4)\n"
            "assert t.allclose(out['expanded'], expected_values), 'expanded values wrong'\n"
            "assert t.allclose(out['repeated'], expected_values), 'repeated values wrong'\n"
            "\n"
            "# The CRITICAL distinction — strides.\n"
            "assert out['expanded_stride'] == (0, 1), (\n"
            "    f'expand-broadcast must have stride 0 on the new axis. Got {out[\"expanded_stride\"]}. '\n"
            "    'If you got nonzero stride you accidentally called .contiguous() or .repeat() — '\n"
            "    'the whole point is that .expand() preserves the stride-0 view.'\n"
            ")\n"
            "assert out['repeated_stride'] == (3, 1), (\n"
            "    f'repeat() makes a contiguous copy — expected stride (3, 1) for shape (4, 3), got {out[\"repeated_stride\"]}'\n"
            ")\n"
            "\n"
            "# Storage sharing.\n"
            "assert out['expanded_shares_storage'] is True, (\n"
            "    'expand() must share storage with x — it is a view. If False, you called .clone() or .contiguous().'\n"
            ")\n"
            "assert out['repeated_shares_storage'] is False, (\n"
            "    'repeat() always copies. If True, you returned the original x somewhere.'\n"
            ")\n"
            "\n"
            "# Larger tensor — confirm memory savings claim holds.\n"
            "big = t.randn(1024, generator=t.Generator().manual_seed(0))\n"
            "big_out = ex1_classify_broadcasts(big)\n"
            "assert big_out['expanded_stride'] == (0, 1)\n"
            "assert big_out['repeated_stride'] == (1024, 1)\n"
            "# The expanded view's storage size equals x's storage size — the (4, 1024) tensor\n"
            "# is virtual, no extra memory was allocated for the broadcast axis.\n"
            "assert big_out['expanded'].untyped_storage().size() == big.untyped_storage().size(), (\n"
            "    'expanded view should reuse x.storage() — same byte count. If different, you copied.'\n"
            ")"
        ),
        "solution_body": (
            "def ex1_classify_broadcasts(x: Tensor) -> dict:\n"
            "    N = x.shape[0]\n"
            "    expanded = x.expand(4, N)\n"
            "    repeated = x.repeat(4, 1)\n"
            "    out = {\n"
            "        'expanded': expanded,\n"
            "        'repeated': repeated,\n"
            "        'expanded_stride': tuple(expanded.stride()),\n"
            "        'repeated_stride': tuple(repeated.stride()),\n"
            "        'expanded_shares_storage': expanded.data_ptr() == x.data_ptr(),\n"
            "        'repeated_shares_storage': repeated.data_ptr() == x.data_ptr(),\n"
            "    }\n"
            "    print(f\"  expanded.stride() = {out['expanded_stride']}  (zero on axis 0 → broadcast view)\")\n"
            "    print(f\"  repeated.stride() = {out['repeated_stride']}  (all nonzero → fresh copy)\")\n"
            "    print(f\"  expanded shares storage with x: {out['expanded_shares_storage']}\")\n"
            "    print(f\"  repeated shares storage with x: {out['repeated_shares_storage']}\")\n"
            "    return out"
        ),
        "solution_notes": (
            "**Why stride-0 is enough.** A tensor's `(i, j)` element is found "
            "at byte offset `i * stride[0] + j * stride[1]` from the storage "
            "base. If `stride[0] == 0`, every row index maps to the same "
            "offset — the underlying storage only needs `N` floats to serve a "
            "virtual `(4, N)` tensor.\n\n"
            "**Why you usually can't write to an expanded view.** PyTorch "
            "blocks in-place writes through stride-0 axes precisely because "
            "the write would alias every position simultaneously, which is "
            "almost never the user's intent. `expanded[2, 1] = 99` raises "
            "`RuntimeError: unsupported operation: more than one element of "
            "the written-to tensor refers to a single memory location`.\n\n"
            "**The 32-byte cost vs the 16-KB cost.** For `x` of length 4096 "
            "broadcast to `(32, 4096)`: `.expand()` uses 16 KB total "
            "(unchanged from `x`), `.repeat()` uses 512 KB. In a transformer "
            "with batch-broadcast attention masks, that gap compounds across "
            "every layer."
        ),
    },
    # ============================================================ sqrt-eps-stabilize / ex1
    {
        "atom_id": "sqrt-eps-stabilize",
        "subtopic": "Numerical: sqrt-eps stabilization",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_SQRT_EPS,
        "exercise_index": 1,
        "exercise_title": "rescue a BatchNorm-style normalize from divide-by-zero",
        "slug": "rescue-batchnorm-from-divide-by-zero",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["batchnorm", "stability", "eps", "nan", "inf"],
        "kcs": ["sqrt-eps-inside-not-outside", "naive-normalize-produces-nan"],
        "lo": (
            "Add the `+ eps` inside-sqrt to a naive normalize so it produces "
            "finite output even when the per-channel variance collapses to 0."
        ),
        "prompt_body": (
            "Implement BOTH functions:\n\n"
            "1. `ex1_naive_normalize(x)` — the BROKEN version, no eps:\n"
            "   ```\n"
            "   mean = x.mean(dim=0)\n"
            "   var  = x.var(dim=0, unbiased=False)\n"
            "   return (x - mean) / t.sqrt(var)\n"
            "   ```\n"
            "   This will divide by zero whenever any channel has identical "
            "values across the batch — common in dead-ReLU channels.\n\n"
            "2. `ex1_stable_normalize(x, eps=1e-5)` — the FIXED version. "
            "Identical structure, but with `+ eps` **inside the sqrt**:\n"
            "   `return (x - mean) / t.sqrt(var + eps)`.\n\n"
            "`x` has shape `(B, C)` — batch × channels. Both functions return "
            "`(B, C)` tensors. The naive version will produce `nan` or `inf` "
            "on the test input that has a constant column; the stable version "
            "must produce finite values for both inputs.\n\n"
            "**Don't try to mask the nans after the fact** — the whole point is "
            "to fix the upstream sqrt by putting eps INSIDE it. `t.sqrt(var) + "
            "eps` is the wrong fix (only patches the divide-by-zero, not the "
            "gradient explosion at small var)."
        ),
        "stub": (
            "def ex1_naive_normalize(x: Tensor) -> Tensor:\n"
            '    """Broken normalize — no eps. Will produce nan/inf for constant channels."""\n'
            "    raise NotImplementedError()\n"
            "\n"
            "def ex1_stable_normalize(x: Tensor, eps: float = 1e-5) -> Tensor:\n"
            '    """Fixed normalize — sqrt(var + eps) keeps output finite even with zero variance."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Input with one collapsed channel — column 1 is constant.\n"
            "x = t.tensor([\n"
            "    [1.0, 5.0, 3.0],\n"
            "    [2.0, 5.0, 7.0],\n"
            "    [3.0, 5.0, 1.0],\n"
            "    [4.0, 5.0, 9.0],\n"
            "])\n"
            "# Naive must produce non-finite output on the collapsed column.\n"
            "naive = ex1_naive_normalize(x)\n"
            "assert naive.shape == (4, 3), f'naive shape wrong: {tuple(naive.shape)}'\n"
            "assert not t.isfinite(naive[:, 1]).any(), (\n"
            "    'naive_normalize should produce nan/inf on the constant column — '\n"
            "    'if it does not, you accidentally added eps. The point is to OBSERVE the failure first.'\n"
            ")\n"
            "# Other columns should still be finite (their variance is positive).\n"
            "assert t.isfinite(naive[:, 0]).all(), 'column 0 should be finite — it has nonzero variance'\n"
            "assert t.isfinite(naive[:, 2]).all(), 'column 2 should be finite — it has nonzero variance'\n"
            "\n"
            "# Stable must produce all-finite output.\n"
            "stable = ex1_stable_normalize(x)\n"
            "assert stable.shape == (4, 3), f'stable shape wrong: {tuple(stable.shape)}'\n"
            "assert t.isfinite(stable).all(), (\n"
            "    f'stable_normalize must produce all finite values. Found non-finite at indices: '\n"
            "    f'{(~t.isfinite(stable)).nonzero()}'\n"
            ")\n"
            "# On the constant column, (x - mean) is 0 → (0) / sqrt(0 + eps) = 0 exactly.\n"
            "assert t.allclose(stable[:, 1], t.zeros(4), atol=1e-5), (\n"
            "    f'constant column should normalize to all zeros, got {stable[:, 1]}'\n"
            ")\n"
            "# On the non-constant columns, output should have mean ~0, std ~1 (since variance is large vs eps).\n"
            "for col in [0, 2]:\n"
            "    col_mean = stable[:, col].mean().item()\n"
            "    col_std = stable[:, col].std(unbiased=False).item()\n"
            "    assert abs(col_mean) < 1e-5, f'col {col} should have ~0 mean, got {col_mean}'\n"
            "    assert abs(col_std - 1.0) < 1e-3, f'col {col} should have ~1 std, got {col_std}'\n"
            "\n"
            "# eps placement check — sabotage detection. If you wrote sqrt(var) + eps instead of sqrt(var + eps),\n"
            "# the output on the constant column would still be (x - mean) / eps = 0 / eps = 0, so this test\n"
            "# alone can't distinguish. But on a CLOSE-TO-zero variance, the two diverge — sqrt(0 + 1e-5) ≈ 0.003\n"
            "# while sqrt(0) + 1e-5 ≈ 1e-5, a 300x difference. Verify with a deliberately tiny-variance column.\n"
            "x_tiny = t.tensor([[1.0, 1.0 + 1e-9], [1.0, 1.0 - 1e-9]])  # variance ~ 1e-18\n"
            "stable_tiny = ex1_stable_normalize(x_tiny, eps=1e-5)\n"
            "assert t.isfinite(stable_tiny).all(), 'must stay finite on tiny-variance input'\n"
            "# With eps INSIDE sqrt: divisor ≈ sqrt(1e-5) ≈ 0.003. Output magnitude ~ 3e-7.\n"
            "# With eps OUTSIDE: divisor ≈ 1e-5. Output magnitude ~ 1e-4.\n"
            "# We expect the inside-sqrt form: output magnitude < 1e-5.\n"
            "max_mag = stable_tiny[:, 1].abs().max().item()\n"
            "assert max_mag < 1e-5, (\n"
            "    f'eps must go INSIDE the sqrt: sqrt(var + eps). '\n"
            "    f'Your output magnitude {max_mag:.2e} suggests you wrote sqrt(var) + eps instead, '\n"
            "    f'which is the wrong fix — it leaves the gradient explosion near zero variance.'\n"
            ")"
        ),
        "solution_body": (
            "def ex1_naive_normalize(x: Tensor) -> Tensor:\n"
            "    mean = x.mean(dim=0)\n"
            "    var = x.var(dim=0, unbiased=False)\n"
            "    return (x - mean) / t.sqrt(var)\n"
            "\n"
            "def ex1_stable_normalize(x: Tensor, eps: float = 1e-5) -> Tensor:\n"
            "    mean = x.mean(dim=0)\n"
            "    var = x.var(dim=0, unbiased=False)\n"
            "    return (x - mean) / t.sqrt(var + eps)"
        ),
        "solution_notes": (
            "**Why the gradient form matters too.** `d/dx sqrt(x) = 1 / (2 * "
            "sqrt(x))`. As `x → 0`, the gradient diverges. With `sqrt(var + "
            "eps)` the derivative is bounded by `1 / (2 * sqrt(eps))` — a "
            "large number, but finite. With `sqrt(var) + eps`, the gradient "
            "is `1 / (2 * sqrt(var))` — still divergent. Only the inside-sqrt "
            "form stabilizes BOTH the forward divide AND the backward "
            "gradient.\n\n"
            "**Why `var(unbiased=False)`.** BatchNorm uses the biased "
            "estimator (`/N` not `/(N-1)`) because the per-batch statistics "
            "are not samples of a population — they ARE the data we're "
            "normalizing. The unbiased correction is for inferring a "
            "population variance from a sample, which is a different "
            "statistical question.\n\n"
            "**What eps to pick.** PyTorch's `nn.BatchNorm2d` defaults to "
            "`1e-5`. `nn.LayerNorm` defaults to `1e-5`. RMSNorm typically "
            "uses `1e-6`. Adam's denominator uses `1e-8`. The order of "
            "magnitude matters more than the exact value — `1e-3` would "
            "audibly shift small-variance channels, `1e-12` is rounded to 0 "
            "in float32."
        ),
    },
    # ============================================================ kaiming-uniform-init / ex1
    {
        "atom_id": "kaiming-uniform-init",
        "subtopic": "Init: Kaiming uniform",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_KAIMING,
        "exercise_index": 1,
        "exercise_title": "build a Linear with Kaiming-uniform init + histogram visualization",
        "slug": "kaiming-uniform-linear-init-with-histogram",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["kaiming", "init", "uniform", "fan_in", "histogram"],
        "kcs": ["kaiming-uniform-bound-formula", "init-weight-uniform-in-place"],
        "lo": (
            "Build a Linear layer whose weight is initialized from Uniform(-1/"
            "sqrt(fan_in), +1/sqrt(fan_in)) and confirm the empirical "
            "distribution matches the theoretical bounds via a histogram."
        ),
        "prompt_body": (
            "Implement `ex1_kaiming_linear(in_features, out_features)` to "
            "build a Linear-style Module whose weight is initialized using "
            "PyTorch's default Kaiming-uniform formula:\n\n"
            "1. Class `KaimingLinear(t.nn.Module)`:\n"
            "   - `__init__(self, in_features, out_features)`:\n"
            "     - `super().__init__()`.\n"
            "     - Store `self.in_features` and `self.out_features`.\n"
            "     - Compute `bound = 1.0 / math.sqrt(in_features)`.\n"
            "     - Sample a weight init via `w_init = t.empty(out_features, "
            "in_features).uniform_(-bound, +bound)` and wrap it: "
            "`self.weight = t.nn.Parameter(w_init)`.\n"
            "     - Same for bias: `b_init = t.empty(out_features).uniform_("
            "-bound, +bound)`, then `self.bias = t.nn.Parameter(b_init)`.\n"
            "   - `forward(self, x): return x @ self.weight.T + self.bias`.\n"
            "2. Return an instance of `KaimingLinear` from `ex1_kaiming_linear(...)`.\n\n"
            "**Why this exact formula.** PyTorch's `nn.Linear.reset_parameters` "
            "calls `nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))`, "
            "which expands to `Uniform(-bound, +bound)` with `bound = sqrt(6 "
            "/ ((1 + 5) * fan_in)) = 1 / sqrt(fan_in)`. The bias gets the "
            "same bound for backward-pass variance preservation.\n\n"
            "**Use `.uniform_(low, high)` (trailing underscore = in place).** "
            "Sample into a plain tensor FIRST, then wrap in `nn.Parameter`. "
            "Calling `.uniform_` directly on a leaf Parameter with "
            "`requires_grad=True` raises `RuntimeError: a leaf Variable that "
            "requires grad is being used in an in-place operation` — you'd "
            "need a `with t.no_grad():` block to do it that way. The "
            "sample-then-wrap pattern is cleaner.\n\n"
            "The visualization plots a histogram of the weight values against "
            "the theoretical `[-bound, +bound]` rectangle so you can confirm "
            "the distribution actually IS uniform on that interval."
        ),
        "stub": (
            "def ex1_kaiming_linear(in_features: int, out_features: int):\n"
            '    """Return a Linear-style Module with Kaiming-uniform weight + bias init."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import math\n"
            "\n"
            "# Build with a LARGE in_features so the histogram is statistically meaningful.\n"
            "in_f, out_f = 256, 64\n"
            "t.manual_seed(0)\n"
            "mod = ex1_kaiming_linear(in_f, out_f)\n"
            "assert isinstance(mod, t.nn.Module)\n"
            "\n"
            "# Both weight + bias must be registered as Parameters.\n"
            "params = dict(mod.named_parameters())\n"
            "assert set(params.keys()) == {'weight', 'bias'}, f'expected weight+bias, got {set(params.keys())}'\n"
            "assert isinstance(params['weight'], t.nn.Parameter), 'weight must remain an nn.Parameter (use in-place .uniform_)'\n"
            "assert isinstance(params['bias'], t.nn.Parameter), 'bias must remain an nn.Parameter (use in-place .uniform_)'\n"
            "assert params['weight'].shape == (out_f, in_f), f'weight shape wrong: {tuple(params[\"weight\"].shape)}'\n"
            "assert params['bias'].shape == (out_f,), f'bias shape wrong: {tuple(params[\"bias\"].shape)}'\n"
            "\n"
            "# Distribution check — weights must lie in [-bound, +bound].\n"
            "bound = 1.0 / math.sqrt(in_f)\n"
            "w = params['weight'].detach()\n"
            "assert w.min().item() >= -bound - 1e-7, (\n"
            "    f'weight min {w.min().item():.6f} below theoretical bound -{bound:.6f}. '\n"
            "    'Wrong distribution — should be Uniform(-1/sqrt(fan_in), +1/sqrt(fan_in)).'\n"
            ")\n"
            "assert w.max().item() <= bound + 1e-7, (\n"
            "    f'weight max {w.max().item():.6f} above theoretical bound +{bound:.6f}.'\n"
            ")\n"
            "# Bias same bound.\n"
            "b = params['bias'].detach()\n"
            "assert b.min().item() >= -bound - 1e-7\n"
            "assert b.max().item() <= bound + 1e-7\n"
            "\n"
            "# Distribution shape check — for uniform on [-bound, bound], theoretical std = bound / sqrt(3).\n"
            "expected_std = bound / math.sqrt(3.0)\n"
            "actual_std = w.std(unbiased=False).item()\n"
            "assert abs(actual_std - expected_std) < 0.15 * expected_std, (\n"
            "    f'weight std {actual_std:.5f} differs >15% from theoretical {expected_std:.5f} '\n"
            "    f'for Uniform(-{bound:.5f}, +{bound:.5f}). Did you use Normal instead of Uniform?'\n"
            ")\n"
            "# Mean should be close to 0 for symmetric uniform.\n"
            "actual_mean = w.mean().item()\n"
            "assert abs(actual_mean) < 0.05 * bound, (\n"
            "    f'weight mean {actual_mean:.5f} too far from 0 for a symmetric Uniform(-bound, +bound)'\n"
            ")\n"
            "\n"
            "# Forward must work — shape check.\n"
            "x = t.randn(8, in_f, generator=t.Generator().manual_seed(0))\n"
            "y = mod(x)\n"
            "assert y.shape == (8, out_f), f'forward shape wrong: {tuple(y.shape)}'\n"
            "\n"
            "# --- Histogram visualization vs theoretical bounds ---\n"
            "fig, ax = plt.subplots(figsize=(7, 3))\n"
            "ax.hist(w.flatten().numpy(), bins=50, density=True, color='steelblue', edgecolor='black', alpha=0.75)\n"
            "# Theoretical uniform PDF height = 1 / (2 * bound)\n"
            "pdf_height = 1.0 / (2 * bound)\n"
            "ax.hlines(pdf_height, -bound, +bound, color='red', linewidth=2, label=f'Uniform(±{bound:.4f}) PDF')\n"
            "ax.axvline(-bound, color='red', linestyle='--', alpha=0.5)\n"
            "ax.axvline(+bound, color='red', linestyle='--', alpha=0.5)\n"
            "ax.set_title(f'Kaiming-uniform weight init — in_features={in_f}, bound=1/sqrt({in_f})={bound:.5f}')\n"
            "ax.set_xlabel('weight value')\n"
            "ax.set_ylabel('density')\n"
            "ax.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex1_kaiming_linear(in_features: int, out_features: int):\n"
            "    import math\n"
            "    class KaimingLinear(t.nn.Module):\n"
            "        def __init__(self, in_features, out_features):\n"
            "            super().__init__()\n"
            "            self.in_features = in_features\n"
            "            self.out_features = out_features\n"
            "            bound = 1.0 / math.sqrt(in_features)\n"
            "            # Sample into a plain tensor (no requires_grad) so .uniform_ is legal,\n"
            "            # then wrap in nn.Parameter — same end state as initializing in place\n"
            "            # on the leaf Parameter under torch.no_grad().\n"
            "            w_init = t.empty(out_features, in_features).uniform_(-bound, +bound)\n"
            "            self.weight = t.nn.Parameter(w_init)\n"
            "            b_init = t.empty(out_features).uniform_(-bound, +bound)\n"
            "            self.bias = t.nn.Parameter(b_init)\n"
            "        def forward(self, x):\n"
            "            return x @ self.weight.T + self.bias\n"
            "    return KaimingLinear(in_features, out_features)"
        ),
        "solution_notes": (
            "**Why `t.empty` not `t.zeros` or `t.randn`.** `t.empty` allocates "
            "uninitialized memory (garbage values) which we immediately "
            "overwrite with `.uniform_`. Allocating with `t.zeros` would "
            "waste a write; allocating with `t.randn` would waste a "
            "different write AND set the wrong distribution before we "
            "fix it.\n\n"
            "**Why in-place `.uniform_` not `t.nn.Parameter(t.rand(...) * "
            "scale)`.** Both work for the math, but the in-place form is "
            "what PyTorch's `reset_parameters` does, and it's the idiom "
            "every reviewer expects. Reassigning a Parameter slot via "
            "`self.weight = t.nn.Parameter(new_tensor)` works but signals "
            "to readers that you don't know about in-place init.\n\n"
            "**The default uses `a=sqrt(5)`, which is a historical "
            "accident.** Modern He/Kaiming for ReLU uses `a=0`, which "
            "gives bound `sqrt(6 / fan_in)` — about 2.45x larger than "
            "PyTorch's default `1/sqrt(fan_in)`. For deep ReLU networks, "
            "the `a=0` variant trains faster; the `a=sqrt(5)` default "
            "exists for legacy compatibility with old PyTorch checkpoints. "
            "Almost every modern repo overrides it."
        ),
        "extra_imports": ["import math", "import matplotlib.pyplot as plt"],
    },
    # ============================================================ device-consistent-construct / ex1
    {
        "atom_id": "device-consistent-construct",
        "subtopic": "PyTorch: Device-consistent tensor construction",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_DEVICE_CONSISTENT,
        "exercise_index": 1,
        "exercise_title": "build a Module that allocates scratch tensors with the right device + dtype",
        "slug": "device-consistent-scratch-allocation",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["device", "dtype", "zeros", "scratch", "moveable-module"],
        "kcs": ["torch-zeros-device-dtype-kwargs", "module-moves-with-to-device"],
        "lo": (
            "Allocate a scratch tensor inside forward() with device=x.device + "
            "dtype=x.dtype kwargs and confirm the Module survives a .to(dtype) "
            "round-trip without spurious upcasts."
        ),
        "prompt_body": (
            "Implement `ex1_residual_accumulator()` — a Module that adds `x` "
            "to a freshly-allocated zero buffer of the same shape, device, "
            "and dtype:\n\n"
            "1. Class `ResidualAccumulator(t.nn.Module)`:\n"
            "   - No `__init__` needed (stateless).\n"
            "   - `forward(self, x: Tensor) -> Tensor`:\n"
            "     - Allocate `buf = t.zeros(x.shape, device=x.device, dtype=x.dtype)` — "
            "**threading BOTH `device` AND `dtype` from `x`**.\n"
            "     - Return `buf + x` (semantically just `x`, but routed "
            "through the scratch tensor so we can audit the allocation).\n"
            "2. Return an instance of `ResidualAccumulator` from `ex1_residual_accumulator()`.\n\n"
            "**Why this isn't a no-op.** The test calls `module(x)` with `x` "
            "on different dtypes (`float32`, `float64`, `bfloat16`) and "
            "asserts the OUTPUT dtype matches `x.dtype` EXACTLY — proving "
            "you didn't accidentally allocate `buf` as default `float32` and "
            "trigger an upcast.\n\n"
            "**The wrong pattern** (`t.zeros(x.shape).to(x.device)`) would:\n"
            "- allocate on CPU first, then transfer (wasted host alloc + "
            "host→device copy);\n"
            "- default to `float32` regardless of `x.dtype`, causing the "
            "addition to upcast `x` if it was `float16`/`bfloat16`/`float64`.\n\n"
            "Bonus: try `torch.zeros_like(x)` as a one-liner alternative — it "
            "automatically inherits device + dtype + memory-layout from `x`. "
            "But for this drill, write out the explicit `device=` + `dtype=` "
            "kwargs so you SEE the pattern."
        ),
        "stub": (
            "def ex1_residual_accumulator():\n"
            '    """Return a Module that allocates a zero scratch tensor with device + dtype matching its input."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "mod = ex1_residual_accumulator()\n"
            "assert isinstance(mod, t.nn.Module)\n"
            "\n"
            "# Test 1 — float32 (default).\n"
            "x32 = t.randn(3, 4)\n"
            "y32 = mod(x32)\n"
            "assert y32.shape == x32.shape\n"
            "assert y32.dtype == t.float32, f'expected float32, got {y32.dtype}'\n"
            "assert t.allclose(y32, x32), 'buf + x should equal x when buf is zeros'\n"
            "\n"
            "# Test 2 — float64. The buf MUST be allocated as float64, not default float32.\n"
            "x64 = t.randn(3, 4).double()\n"
            "y64 = mod(x64)\n"
            "assert y64.dtype == t.float64, (\n"
            "    f'expected float64, got {y64.dtype}. '\n"
            "    'You forgot dtype=x.dtype — t.zeros defaults to float32, then x got upcast/downcast.'\n"
            ")\n"
            "assert t.allclose(y64, x64)\n"
            "\n"
            "# Test 3 — bfloat16 (the mixed-precision case).\n"
            "x_bf = t.randn(3, 4).to(t.bfloat16)\n"
            "y_bf = mod(x_bf)\n"
            "assert y_bf.dtype == t.bfloat16, (\n"
            "    f'expected bfloat16, got {y_bf.dtype}. '\n"
            "    'bfloat16 + float32 promotes to float32 — defeats mixed precision.'\n"
            ")\n"
            "assert t.allclose(y_bf, x_bf, atol=1e-2)  # bfloat16 has low precision\n"
            "\n"
            "# Test 4 — Module should survive a .to(dtype) on the whole Module.\n"
            "# (Stateless Module → .to() is a no-op for params, but a real moveable Module\n"
            "#  must still produce the right dtype on forward.)\n"
            "mod.to(t.float64)\n"
            "y_promoted = mod(x32)  # input still float32 — buf still allocated as input.dtype\n"
            "assert y_promoted.dtype == t.float32, (\n"
            "    f'forward output dtype should track INPUT dtype (x32 is float32), got {y_promoted.dtype}'\n"
            ")\n"
            "\n"
            "# Test 5 — device propagation. We only have CPU here, but the test asserts the buf\n"
            "# .device matches x.device — proves the wiring is right even without a GPU.\n"
            "x_cpu = t.randn(5, 5)\n"
            "y_cpu = mod(x_cpu)\n"
            "assert y_cpu.device == x_cpu.device, f'device mismatch: {y_cpu.device} vs {x_cpu.device}'\n"
            "\n"
            "# Test 6 — sabotage detection. Allocate a tensor that would CRASH if buf were\n"
            "# allocated with default dtype on CPU then transferred. We can't test GPU here,\n"
            "# but we CAN test that the user used `device=x.device` not `.to(x.device)`.\n"
            "# We do this by intercepting torch.zeros and checking the kwargs received.\n"
            "import unittest.mock as _mock\n"
            "real_zeros = t.zeros\n"
            "captured = {}\n"
            "def _spy(*args, **kwargs):\n"
            "    captured['args'] = args\n"
            "    captured['kwargs'] = dict(kwargs)\n"
            "    return real_zeros(*args, **kwargs)\n"
            "with _mock.patch.object(t, 'zeros', side_effect=_spy):\n"
            "    _ = mod(x64)\n"
            "assert 'device' in captured['kwargs'], (\n"
            "    'You must pass device=x.device to torch.zeros, not call .to(device) afterward. '\n"
            "    f'Captured zeros() kwargs: {captured[\"kwargs\"]}'\n"
            ")\n"
            "assert 'dtype' in captured['kwargs'], (\n"
            "    'You must pass dtype=x.dtype to torch.zeros. '\n"
            "    f'Captured zeros() kwargs: {captured[\"kwargs\"]}'\n"
            ")"
        ),
        "solution_body": (
            "def ex1_residual_accumulator():\n"
            "    class ResidualAccumulator(t.nn.Module):\n"
            "        def forward(self, x: Tensor) -> Tensor:\n"
            "            buf = t.zeros(x.shape, device=x.device, dtype=x.dtype)\n"
            "            return buf + x\n"
            "    return ResidualAccumulator()"
        ),
        "solution_notes": (
            "**Why `device=x.device` not `.to(x.device)`.** `t.zeros(shape, "
            "device='cuda:0')` calls the CUDA allocator directly — one "
            "device-side allocation, zero data movement. `t.zeros(shape)."
            "to('cuda:0')` does a CPU malloc + zero-fill + host→device copy "
            "+ CPU dealloc — three operations and a host-device sync, every "
            "single forward pass.\n\n"
            "**Why `dtype=x.dtype` matters in mixed precision.** AMP "
            "(automatic mixed precision) casts the input to `float16` or "
            "`bfloat16` to halve memory and ~double throughput. If your "
            "scratch tensor allocates as `float32`, the addition promotes "
            "the whole expression back to `float32` — silently undoing the "
            "AMP savings. The bug is invisible (output still numerically "
            "correct) but throughput regresses without explanation.\n\n"
            "**`zeros_like(x)` is the one-liner.** `t.zeros_like(x)` "
            "inherits shape + device + dtype + layout from `x`. Same for "
            "`ones_like` / `empty_like` / `full_like`. Use them whenever the "
            "new tensor shape matches an existing one. The explicit "
            "`device=`+`dtype=` form is for when the shape differs (e.g. "
            "allocating an output buffer that's the prefix-shape of `x`)."
        ),
    },
    # ============================================================ conditional-hparam-branch / ex1
    {
        "atom_id": "conditional-hparam-branch",
        "subtopic": "PyTorch: Conditional hparam branch",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_COND_HPARAM,
        "exercise_index": 1,
        "exercise_title": "Linear with optional bias gated by use_bias flag",
        "slug": "linear-with-optional-bias-flag",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["bias", "optional", "conditional", "if-branch", "use_bias"],
        "kcs": ["conditional-param-register", "forward-branch-on-none"],
        "lo": (
            "Define a Module whose `__init__` registers `self.bias` as an "
            "nn.Parameter when use_bias=True and as `None` otherwise, with "
            "the matching forward-pass `is not None` branch."
        ),
        "prompt_body": (
            "Implement `OptionalBiasLinear` — a Linear-style Module where "
            "the bias is optional based on a constructor flag:\n\n"
            "1. Class `OptionalBiasLinear(t.nn.Module)`:\n"
            "   - `__init__(self, in_features, out_features, use_bias=True)`:\n"
            "     - Call `super().__init__()`.\n"
            "     - Store `self.in_features`, `self.out_features`, `self.use_bias`.\n"
            "     - Create `self.weight = t.nn.Parameter(t.zeros(out_features, "
            "in_features))` (zero-init so the math is predictable in tests).\n"
            "     - **Conditional branch on `use_bias`:**\n"
            "       - If `use_bias`: `self.bias = t.nn.Parameter(t.zeros("
            "out_features))`.\n"
            "       - Else: `self.bias = None`.  # NOT `del` or skipped\n"
            "   - `forward(self, x: Tensor) -> Tensor`:\n"
            "     - `out = x @ self.weight.T`\n"
            "     - **Conditional branch:** `if self.bias is not None: out = "
            "out + self.bias`.\n"
            "     - Return `out`.\n"
            "2. Return an instance from `ex1_optional_bias_linear(in_features, "
            "out_features, use_bias)`.\n\n"
            "**The two halves are coupled.** Setting `self.bias = None` "
            "(rather than skipping the assignment) means `forward` can "
            "ALWAYS reference `self.bias` — the `is not None` guard then "
            "handles the absent case. If you skipped the assignment, the "
            "forward branch would raise `AttributeError` at the worst "
            "possible time (mid-training).\n\n"
            "The test asserts: (a) with `use_bias=True`, both weight and bias "
            "show up in `.parameters()`; (b) with `use_bias=False`, only "
            "weight appears AND `self.bias is None`; (c) forward output "
            "matches the right algebraic expression in both modes."
        ),
        "stub": (
            "def ex1_optional_bias_linear(in_features: int, out_features: int, use_bias: bool):\n"
            '    """Return an OptionalBiasLinear instance with bias registered iff use_bias=True."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Case 1 — use_bias=True. Both weight and bias must be Parameters.\n"
            "mod_with = ex1_optional_bias_linear(in_features=3, out_features=4, use_bias=True)\n"
            "assert isinstance(mod_with, t.nn.Module)\n"
            "params_with = dict(mod_with.named_parameters())\n"
            "assert set(params_with.keys()) == {'weight', 'bias'}, (\n"
            "    f'use_bias=True should register weight+bias, got {set(params_with.keys())}'\n"
            ")\n"
            "assert isinstance(params_with['bias'], t.nn.Parameter)\n"
            "assert params_with['bias'].shape == (4,)\n"
            "assert mod_with.bias is not None, 'mod.bias should be a Parameter when use_bias=True'\n"
            "\n"
            "# Case 2 — use_bias=False. Only weight; self.bias is None.\n"
            "mod_without = ex1_optional_bias_linear(in_features=3, out_features=4, use_bias=False)\n"
            "params_without = dict(mod_without.named_parameters())\n"
            "assert set(params_without.keys()) == {'weight'}, (\n"
            "    f'use_bias=False should register weight only, got {set(params_without.keys())}'\n"
            ")\n"
            "assert mod_without.bias is None, (\n"
            "    'when use_bias=False you must set self.bias = None, NOT skip the assignment. '\n"
            "    f'Got: {mod_without.bias!r}'\n"
            ")\n"
            "# .parameters() must be exactly length 1 (just the weight).\n"
            "assert len(list(mod_without.parameters())) == 1, (\n"
            "    f'expected 1 parameter (weight only), got {len(list(mod_without.parameters()))}'\n"
            ")\n"
            "\n"
            "# Case 3 — forward correctness. Hand-set weights so we can predict output.\n"
            "with t.no_grad():\n"
            "    mod_with.weight.copy_(t.ones(4, 3))\n"
            "    mod_with.bias.copy_(t.tensor([10., 20., 30., 40.]))\n"
            "    mod_without.weight.copy_(t.ones(4, 3))\n"
            "\n"
            "x = t.tensor([1.0, 2.0, 3.0])  # sum = 6\n"
            "y_with = mod_with(x)\n"
            "# expected: x @ weight.T + bias = [6, 6, 6, 6] + [10, 20, 30, 40] = [16, 26, 36, 46]\n"
            "expected_with = t.tensor([16.0, 26.0, 36.0, 46.0])\n"
            "assert t.allclose(y_with, expected_with), f'with-bias forward wrong: got {y_with}, expected {expected_with}'\n"
            "\n"
            "y_without = mod_without(x)\n"
            "# expected: x @ weight.T = [6, 6, 6, 6]\n"
            "expected_without = t.tensor([6.0, 6.0, 6.0, 6.0])\n"
            "assert t.allclose(y_without, expected_without), f'no-bias forward wrong: got {y_without}, expected {expected_without}'\n"
            "\n"
            "# Case 4 — batched forward works in both modes.\n"
            "x_batch = t.randn(5, 3, generator=t.Generator().manual_seed(0))\n"
            "assert mod_with(x_batch).shape == (5, 4)\n"
            "assert mod_without(x_batch).shape == (5, 4)"
        ),
        "solution_body": (
            "def ex1_optional_bias_linear(in_features: int, out_features: int, use_bias: bool):\n"
            "    class OptionalBiasLinear(t.nn.Module):\n"
            "        def __init__(self, in_features, out_features, use_bias=True):\n"
            "            super().__init__()\n"
            "            self.in_features = in_features\n"
            "            self.out_features = out_features\n"
            "            self.use_bias = use_bias\n"
            "            self.weight = t.nn.Parameter(t.zeros(out_features, in_features))\n"
            "            if use_bias:\n"
            "                self.bias = t.nn.Parameter(t.zeros(out_features))\n"
            "            else:\n"
            "                self.bias = None\n"
            "        def forward(self, x: Tensor) -> Tensor:\n"
            "            out = x @ self.weight.T\n"
            "            if self.bias is not None:\n"
            "                out = out + self.bias\n"
            "            return out\n"
            "    return OptionalBiasLinear(in_features, out_features, use_bias)"
        ),
        "solution_notes": (
            "**Why `self.bias = None` and not skipping the assignment.** "
            "`nn.Module.__setattr__` has a special case for `None` — setting "
            "a Parameter slot to `None` un-registers it (or never registers "
            "it) without raising. That's why `mod_without.bias` returns "
            "`None` cleanly and `forward`'s `is not None` guard works.\n\n"
            "**Why `self.bias is not None` and not `self.bias`.** Calling "
            "`bool(tensor)` on a tensor with multiple elements raises "
            "`RuntimeError: Boolean value of Tensor with more than one "
            "element is ambiguous`. The `is not None` check sidesteps the "
            "ambiguity — it's the safe idiom and what `nn.Linear.forward` "
            "actually does.\n\n"
            "**`nn.Identity()` as the alternative.** If the conditional "
            "branch is on a hot path (every forward) you can avoid the `is "
            "not None` check by assigning `self.dropout = nn.Dropout(p) if p "
            "> 0 else nn.Identity()`. Then `forward` is always `x = self."
            "dropout(x)` — `Identity.forward(x)` just returns `x`. Tradeoff: "
            "a tiny per-call overhead for the no-op Module vs. a branch in "
            "Python."
        ),
    },
    # ============================================================ rearrange-as-sequential-layer / ex1
    {
        "atom_id": "rearrange-as-sequential-layer",
        "subtopic": "Einops: Rearrange as nn.Sequential layer",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_REARRANGE_LAYER,
        "exercise_index": 1,
        "exercise_title": "build a Conv-flatten-Linear pipeline with Rearrange layer (no forward boilerplate)",
        "slug": "conv-rearrange-linear-pipeline",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["einops", "Rearrange", "layer", "Sequential", "Flatten"],
        "kcs": ["rearrange-layer-import-path", "rearrange-in-sequential-composes"],
        "lo": (
            "Wrap einops.layers.torch.Rearrange inside an nn.Sequential so a "
            "Conv → ReLU → flatten → Linear pipeline composes without writing "
            "a custom forward."
        ),
        "prompt_body": (
            "Implement `ex1_conv_classifier(in_channels, height, width, "
            "num_classes)` — a minimal image classifier that uses "
            "`einops.layers.torch.Rearrange` as a layer inside "
            "`nn.Sequential`, eliminating the need for a custom `forward()`:\n\n"
            "1. Import `from einops.layers.torch import Rearrange` (capital "
            "R — this is the Module form, not the `einops.rearrange` "
            "function).\n"
            "2. Build:\n"
            "   ```\n"
            "   nn.Sequential(\n"
            "       nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),  # (B, 8, H, W)\n"
            "       nn.ReLU(),\n"
            "       Rearrange('b c h w -> b (c h w)'),                     # (B, 8*H*W)\n"
            "       nn.Linear(8 * height * width, num_classes),            # (B, num_classes)\n"
            "   )\n"
            "   ```\n"
            "3. Return the `nn.Sequential` instance directly — DO NOT wrap it "
            "in your own Module subclass. The point of this drill is that "
            "`Rearrange` makes the wrapper unnecessary.\n\n"
            "Input shape: `(B, in_channels, height, width)`. Output shape: "
            "`(B, num_classes)`.\n\n"
            "**Why this beats `nn.Flatten()`.** Flatten's signature requires "
            "you to remember `start_dim=1, end_dim=-1` to keep the batch "
            "axis. `Rearrange('b c h w -> b (c h w)')` puts the shape "
            "transformation in the source — anyone reading the code knows "
            "exactly what shape comes out without consulting docs.\n\n"
            "**Critical import path.** `einops.rearrange` is the FUNCTION "
            "(use inside `forward`). `einops.layers.torch.Rearrange` is the "
            "MODULE (use inside `nn.Sequential`). They share the same string "
            "grammar; only the wrapping differs."
        ),
        "stub": (
            "def ex1_conv_classifier(in_channels: int, height: int, width: int, num_classes: int):\n"
            '    """Return an nn.Sequential pipeline using einops.layers.torch.Rearrange as the flatten layer."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from einops.layers.torch import Rearrange\n"
            "\n"
            "model = ex1_conv_classifier(in_channels=3, height=8, width=8, num_classes=10)\n"
            "\n"
            "# The model must be an nn.Sequential — NOT a custom Module wrapping one.\n"
            "assert isinstance(model, t.nn.Sequential), (\n"
            "    f'expected nn.Sequential, got {type(model).__name__}. '\n"
            "    'The whole point is that Rearrange-as-layer means you do not need a custom Module.'\n"
            ")\n"
            "\n"
            "# Sequential should contain exactly 4 layers in the right order.\n"
            "assert len(model) == 4, f'expected 4 layers (Conv, ReLU, Rearrange, Linear), got {len(model)}'\n"
            "assert isinstance(model[0], t.nn.Conv2d), f'layer 0 should be Conv2d, got {type(model[0]).__name__}'\n"
            "assert isinstance(model[1], t.nn.ReLU), f'layer 1 should be ReLU, got {type(model[1]).__name__}'\n"
            "assert isinstance(model[2], Rearrange), (\n"
            "    f'layer 2 should be einops.layers.torch.Rearrange, got {type(model[2]).__name__}. '\n"
            "    'Did you use nn.Flatten() instead? This drill specifically tests the Rearrange-as-layer pattern.'\n"
            ")\n"
            "assert isinstance(model[3], t.nn.Linear), f'layer 3 should be Linear, got {type(model[3]).__name__}'\n"
            "\n"
            "# Layer dimensions.\n"
            "assert model[0].in_channels == 3\n"
            "assert model[0].out_channels == 8\n"
            "assert model[3].in_features == 8 * 8 * 8  # 8 channels * 8 H * 8 W\n"
            "assert model[3].out_features == 10\n"
            "\n"
            "# Forward — shape check.\n"
            "x = t.randn(4, 3, 8, 8, generator=t.Generator().manual_seed(0))\n"
            "y = model(x)\n"
            "assert y.shape == (4, 10), f'expected (4, 10), got {tuple(y.shape)}'\n"
            "\n"
            "# Verify the Rearrange step DID flatten the conv output (not just transpose).\n"
            "# Run partial pipeline up through the Rearrange and check the shape.\n"
            "intermediate = model[0:3](x)  # Conv → ReLU → Rearrange\n"
            "assert intermediate.shape == (4, 8 * 8 * 8), (\n"
            "    f'after Rearrange the shape should be (B, c*h*w) = (4, 512), got {tuple(intermediate.shape)}. '\n"
            "    'Check your Rearrange string — should be \"b c h w -> b (c h w)\".'\n"
            ")\n"
            "\n"
            "# Test with different spatial dims to confirm the Rearrange string generalizes.\n"
            "model2 = ex1_conv_classifier(in_channels=1, height=4, width=12, num_classes=3)\n"
            "x2 = t.randn(2, 1, 4, 12, generator=t.Generator().manual_seed(1))\n"
            "y2 = model2(x2)\n"
            "assert y2.shape == (2, 3), f'second config wrong: {tuple(y2.shape)}'\n"
            "\n"
            "# Confirm there's no custom Module wrapping — parameters should belong to the\n"
            "# top-level Sequential, not a nested 'model.net.*' path.\n"
            "named = dict(model.named_parameters())\n"
            "# Expected: '0.weight', '0.bias' (Conv2d), '3.weight', '3.bias' (Linear). ReLU + Rearrange have no params.\n"
            "expected_param_names = {'0.weight', '0.bias', '3.weight', '3.bias'}\n"
            "assert set(named.keys()) == expected_param_names, (\n"
            "    f'expected param names {expected_param_names}, got {set(named.keys())}. '\n"
            "    'If you see net.0.weight etc., you wrapped Sequential inside a custom Module — undo that.'\n"
            ")"
        ),
        "solution_body": (
            "def ex1_conv_classifier(in_channels: int, height: int, width: int, num_classes: int):\n"
            "    from einops.layers.torch import Rearrange\n"
            "    return t.nn.Sequential(\n"
            "        t.nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),\n"
            "        t.nn.ReLU(),\n"
            "        Rearrange('b c h w -> b (c h w)'),\n"
            "        t.nn.Linear(8 * height * width, num_classes),\n"
            "    )"
        ),
        "solution_notes": (
            "**Why `Rearrange` is a Module.** "
            "`einops.layers.torch.Rearrange.__init__` stores the einops "
            "pattern string; its `forward` calls `einops.rearrange(x, "
            "self.pattern)`. Because it subclasses `nn.Module`, `nn."
            "Sequential` accepts it and auto-pipes the output. Zero "
            "parameters, no `__init__` work needed by you.\n\n"
            "**The full sibling set in `einops.layers.torch`.** "
            "`Rearrange` (the layer form of `einops.rearrange`), `Reduce` "
            "(the layer form of `einops.reduce` — handy for global "
            "average pool inside Sequential), and `EinMix` (a learnable "
            "Einsum-based linear layer). All three accept the same string "
            "grammar as their function-form counterparts.\n\n"
            "**ARENA's `make_cnn` uses this exact pattern** — Conv → BN → "
            "ReLU stages followed by `Rearrange('b c h w -> b (c h w)')` "
            "and a final Linear. Knowing the Rearrange-as-layer trick is "
            "what lets the whole thing fit in a single `nn.Sequential` "
            "with no custom Module wrapper."
        ),
        "extra_imports": ["from einops.layers.torch import Rearrange"],
    },
    # ============================================================ encoder-decoder-symmetric / ex1
    {
        "atom_id": "encoder-decoder-symmetric",
        "subtopic": "CNN: Encoder-decoder symmetric layout",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_ENC_DEC,
        "exercise_index": 1,
        "exercise_title": "build a tiny autoencoder whose output shape == input shape",
        "slug": "tiny-autoencoder-symmetric-layout",
        "bloom_level": "Create",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["autoencoder", "encoder", "decoder", "symmetric", "upsample"],
        "kcs": ["encoder-downsample-stage", "decoder-upsample-mirror-stage"],
        "lo": (
            "Build a symmetric encoder-decoder Module where each encoder pool "
            "is mirrored by a decoder upsample so input shape == output "
            "shape end-to-end."
        ),
        "prompt_body": (
            "Implement `ex1_tiny_autoencoder(in_channels)` — a minimal but "
            "structurally-correct autoencoder. Spatial dims downsample by 4× "
            "(two pool stages) then upsample back by 4× (two upsample "
            "stages). Channels mirror: `C → 16 → 32` (encoder), `32 → 16 → "
            "C` (decoder).\n\n"
            "1. Class `TinyAutoencoder(t.nn.Module)`:\n"
            "   - `__init__(self, in_channels)`:\n"
            "     - `super().__init__()`.\n"
            "     - `self.encoder = t.nn.Sequential(`\n"
            "       `   t.nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),`\n"
            "       `   t.nn.ReLU(),`\n"
            "       `   t.nn.MaxPool2d(2),                # spatial /= 2`\n"
            "       `   t.nn.Conv2d(16, 32, kernel_size=3, padding=1),`\n"
            "       `   t.nn.ReLU(),`\n"
            "       `   t.nn.MaxPool2d(2),                # spatial /= 2 again`\n"
            "       `)`\n"
            "     - `self.decoder = t.nn.Sequential(`\n"
            "       `   t.nn.Upsample(scale_factor=2),    # spatial *= 2`\n"
            "       `   t.nn.Conv2d(32, 16, kernel_size=3, padding=1),`\n"
            "       `   t.nn.ReLU(),`\n"
            "       `   t.nn.Upsample(scale_factor=2),    # spatial *= 2 again`\n"
            "       `   t.nn.Conv2d(16, in_channels, kernel_size=3, padding=1),`\n"
            "       `)`\n"
            "   - `forward(self, x): return self.decoder(self.encoder(x))`\n"
            "2. Return an instance from `ex1_tiny_autoencoder(in_channels)`.\n\n"
            "**Why no final ReLU in the decoder.** The output is a "
            "reconstruction in pixel space — if your inputs include negative "
            "values (e.g. zero-mean normalized images), a final ReLU clips "
            "them. A sigmoid is appropriate for `[0, 1]` images; raw linear "
            "output is appropriate for normalized images.\n\n"
            "**Why `Upsample + Conv2d` not `ConvTranspose2d`.** ConvTranspose "
            "is learnable upsample in one shot but produces checkerboard "
            "artifacts. Upsample (nearest-neighbor) followed by a regular "
            "Conv2d is artifact-free and what U-Net actually uses.\n\n"
            "**The shape-parity test.** The whole point of symmetric layout: "
            "`model(x).shape == x.shape` for any valid input. The test "
            "checks this on multiple input sizes."
        ),
        "stub": (
            "def ex1_tiny_autoencoder(in_channels: int):\n"
            '    """Return a symmetric encoder-decoder Module: spatial /= 4 then *= 4, channels C → 32 → C."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "mod = ex1_tiny_autoencoder(in_channels=3)\n"
            "assert isinstance(mod, t.nn.Module)\n"
            "\n"
            "# Must have encoder + decoder as named children.\n"
            "kids = dict(mod.named_children())\n"
            "assert 'encoder' in kids, f'expected encoder child, got {list(kids.keys())}'\n"
            "assert 'decoder' in kids, f'expected decoder child, got {list(kids.keys())}'\n"
            "assert isinstance(kids['encoder'], t.nn.Sequential), 'encoder should be nn.Sequential'\n"
            "assert isinstance(kids['decoder'], t.nn.Sequential), 'decoder should be nn.Sequential'\n"
            "\n"
            "# Count pool stages in encoder vs upsample stages in decoder — they MUST match.\n"
            "n_pools = sum(1 for m in kids['encoder'].modules() if isinstance(m, t.nn.MaxPool2d))\n"
            "n_ups = sum(1 for m in kids['decoder'].modules() if isinstance(m, t.nn.Upsample))\n"
            "assert n_pools == n_ups == 2, (\n"
            "    f'encoder pools ({n_pools}) must match decoder upsamples ({n_ups}), both should be 2'\n"
            ")\n"
            "\n"
            "# THE shape-parity test — model(x).shape == x.shape, multiple sizes.\n"
            "for (B, H, W) in [(2, 16, 16), (1, 32, 32), (4, 8, 24), (1, 64, 48)]:\n"
            "    x = t.randn(B, 3, H, W, generator=t.Generator().manual_seed(H + W))\n"
            "    y = mod(x)\n"
            "    assert y.shape == x.shape, (\n"
            "        f'shape parity FAILED for input {tuple(x.shape)}: got {tuple(y.shape)}. '\n"
            "        f'The encoder downsamples by {2**n_pools}x; the decoder must upsample by the SAME factor.'\n"
            "    )\n"
            "\n"
            "# Encoder intermediate shape — confirm spatial dims really do drop by 4x.\n"
            "x_test = t.randn(1, 3, 16, 16)\n"
            "encoded = mod.encoder(x_test)\n"
            "assert encoded.shape == (1, 32, 4, 4), (\n"
            "    f'encoder output shape wrong: {tuple(encoded.shape)}. Expected (1, 32, 4, 4) — '\n"
            "    f'two MaxPool2d(2) stages drop 16→8→4, channels rise 3→16→32.'\n"
            "    )\n"
            "\n"
            "# Channel mirror — encoder's last conv should output 32 channels, decoder's last conv should output 3.\n"
            "encoder_convs = [m for m in kids['encoder'].modules() if isinstance(m, t.nn.Conv2d)]\n"
            "decoder_convs = [m for m in kids['decoder'].modules() if isinstance(m, t.nn.Conv2d)]\n"
            "assert encoder_convs[-1].out_channels == 32, f'encoder last conv should output 32 ch, got {encoder_convs[-1].out_channels}'\n"
            "assert decoder_convs[-1].out_channels == 3, (\n"
            "    f'decoder last conv must output in_channels={3} to mirror back to input shape, '\n"
            "    f'got {decoder_convs[-1].out_channels}'\n"
            ")\n"
            "\n"
            "# Different in_channels (e.g. grayscale).\n"
            "mod_gray = ex1_tiny_autoencoder(in_channels=1)\n"
            "x_gray = t.randn(2, 1, 32, 32)\n"
            "y_gray = mod_gray(x_gray)\n"
            "assert y_gray.shape == (2, 1, 32, 32), f'grayscale autoencoder shape wrong: {tuple(y_gray.shape)}'"
        ),
        "solution_body": (
            "def ex1_tiny_autoencoder(in_channels: int):\n"
            "    class TinyAutoencoder(t.nn.Module):\n"
            "        def __init__(self, in_channels):\n"
            "            super().__init__()\n"
            "            self.encoder = t.nn.Sequential(\n"
            "                t.nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),\n"
            "                t.nn.ReLU(),\n"
            "                t.nn.MaxPool2d(2),\n"
            "                t.nn.Conv2d(16, 32, kernel_size=3, padding=1),\n"
            "                t.nn.ReLU(),\n"
            "                t.nn.MaxPool2d(2),\n"
            "            )\n"
            "            self.decoder = t.nn.Sequential(\n"
            "                t.nn.Upsample(scale_factor=2),\n"
            "                t.nn.Conv2d(32, 16, kernel_size=3, padding=1),\n"
            "                t.nn.ReLU(),\n"
            "                t.nn.Upsample(scale_factor=2),\n"
            "                t.nn.Conv2d(16, in_channels, kernel_size=3, padding=1),\n"
            "            )\n"
            "        def forward(self, x):\n"
            "            return self.decoder(self.encoder(x))\n"
            "    return TinyAutoencoder(in_channels)"
        ),
        "solution_notes": (
            "**Why `padding=1` on every Conv2d.** With `kernel_size=3` and "
            "`padding=1`, spatial dims are preserved by the conv itself — "
            "only the explicit `MaxPool2d` / `Upsample` stages change "
            "spatial dims. This separates the two concerns: convolutions do "
            "feature mixing at fixed resolution, pool/upsample changes "
            "resolution. Without `padding=1`, every conv would shave 2 "
            "pixels off the spatial dims, making the shape arithmetic "
            "miserable.\n\n"
            "**Why MaxPool2d for downsampling.** Three options: strided "
            "Conv, MaxPool, AvgPool. MaxPool is the canonical choice for "
            "early CNN encoders (preserves edges, cheap, no extra "
            "parameters). Strided Conv is what modern architectures use "
            "(more expressive, more parameters). AvgPool is rare in "
            "encoders but common as a final global pool before the "
            "classifier.\n\n"
            "**The skip-connection extension.** This Module is a vanilla "
            "autoencoder — encoder output goes through a bottleneck, "
            "decoder reconstructs from the bottleneck alone. U-Net adds "
            "skip connections: each encoder stage's pre-pool activations "
            "are concatenated to the matching decoder stage's post-upsample "
            "activations. Symmetric layout is what makes those concatenations "
            "shape-compatible.\n\n"
            "**Why a Module subclass here, not just Sequential.** Because "
            "the encoder and decoder are two named sub-pipelines we want "
            "separately introspectable (`mod.encoder`, `mod.decoder`). "
            "A flat Sequential would work for forward but lose the "
            "encoder/decoder labeling — useful for visualizing the latent, "
            "for fine-tuning just the decoder, etc."
        ),
    },
    # ============================================================ loss-item-scalar-extract / ex1
    {
        "atom_id": "loss-item-scalar-extract",
        "subtopic": "PyTorch: loss.item() scalar extract",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_LOSS_ITEM,
        "exercise_index": 1,
        "exercise_title": "distinguish .item() from .detach().cpu() in a training-loop logger",
        "slug": "loss-item-vs-detach-cpu",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["loss", "item", "detach", "logging", "graph-pinning"],
        "kcs": ["loss-item-returns-python-float", "detach-clone-for-buffer"],
        "lo": (
            "Use `.item()` to log a scalar float and `.detach().clone()` to "
            "buffer many step values, then assert that `.item()` returns a "
            "Python float (not a Tensor) and breaks the autograd graph."
        ),
        "prompt_body": (
            "Implement `ex1_train_log(steps)` — a fake training loop that "
            "logs the loss in TWO ways and returns BOTH:\n\n"
            "1. `steps`: int, number of training iterations to simulate.\n"
            "2. Run a synthetic training loop: set `x = t.randn(steps, "
            "requires_grad=False)` as the per-step input. The 'loss' for "
            "step `i` is `loss_i = (x[i] ** 2 + 1.0).requires_grad_(True)` — "
            "a 0-D scalar tensor with a real autograd graph attached (so "
            "`.grad_fn` is not None).\n"
            "3. For each step, populate TWO logs:\n"
            "   - `floats_log[i] = loss_i.item()` — Python float for "
            "wandb-style logging.\n"
            "   - `tensors_log[i] = loss_i.detach().clone()` — detached "
            "0-D tensor for in-memory buffering.\n"
            "4. After the loop, return a dict:\n"
            "   - `'floats_log'`: a Python `list[float]` of length `steps`.\n"
            "   - `'tensors_log'`: a tensor of shape `(steps,)` built via "
            "`t.stack(tensors_log)`.\n"
            "   - `'sample_float_type'`: `type(floats_log[0]).__name__` — "
            "must be `'float'`.\n"
            "   - `'sample_tensor_type'`: `type(tensors_log[0]).__name__` — "
            "must be `'Tensor'`.\n"
            "   - `'sample_tensor_grad_fn'`: `tensors_log[0].grad_fn` — must "
            "be `None` (detached).\n\n"
            "**Why both patterns exist.** `.item()` synchronizes and "
            "extracts ONE value — perfect for wandb / tqdm / print. "
            "`.detach().clone()` keeps the tensor shape, breaks the autograd "
            "graph (so the compute history can be freed), and is the right "
            "choice for buffering many step values for later analysis.\n\n"
            "**The sabotage trap.** Do NOT do `tensors_log[i] = loss_i` "
            "(without detach). That keeps the autograd graph alive across "
            "all `steps` — memory grows linearly, and `tensors_log[0]."
            "grad_fn` would be non-None. The test asserts grad_fn is None "
            "to catch this exact bug."
        ),
        "stub": (
            "def ex1_train_log(steps: int) -> dict:\n"
            '    """Simulate a training loop and return both .item() float log and .detach().clone() tensor log."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "out = ex1_train_log(steps=5)\n"
            "assert isinstance(out, dict)\n"
            "required = {'floats_log', 'tensors_log', 'sample_float_type', 'sample_tensor_type', 'sample_tensor_grad_fn'}\n"
            "assert set(out.keys()) == required, f'expected keys {required}, got {set(out.keys())}'\n"
            "\n"
            "# floats_log must be a Python list of Python floats.\n"
            "fl = out['floats_log']\n"
            "assert isinstance(fl, list), f'floats_log must be a list, got {type(fl).__name__}'\n"
            "assert len(fl) == 5, f'expected 5 entries, got {len(fl)}'\n"
            "for i, v in enumerate(fl):\n"
            "    assert isinstance(v, float), (\n"
            "        f'floats_log[{i}] must be a Python float (from .item()), got {type(v).__name__}. '\n"
            "        'If you got Tensor, you appended the loss without calling .item().'\n"
            "    )\n"
            "assert out['sample_float_type'] == 'float', f'expected \"float\", got {out[\"sample_float_type\"]!r}'\n"
            "\n"
            "# tensors_log must be a stacked tensor of shape (5,).\n"
            "tl = out['tensors_log']\n"
            "assert isinstance(tl, t.Tensor), f'tensors_log must be a Tensor (after t.stack), got {type(tl).__name__}'\n"
            "assert tl.shape == (5,), f'tensors_log shape should be (5,), got {tuple(tl.shape)}'\n"
            "assert out['sample_tensor_type'] == 'Tensor', f'expected \"Tensor\", got {out[\"sample_tensor_type\"]!r}'\n"
            "\n"
            "# THE critical check — the detached tensors must have NO autograd graph.\n"
            "assert out['sample_tensor_grad_fn'] is None, (\n"
            "    f'sample_tensor_grad_fn must be None — you must .detach() before buffering. '\n"
            "    f'Got {out[\"sample_tensor_grad_fn\"]!r}. This is the graph-pinning bug: '\n"
            "    'storing un-detached losses keeps the entire compute graph alive for every step.'\n"
            "    )\n"
            "# tensors_log itself should also have no grad (t.stack of detached tensors).\n"
            "assert tl.grad_fn is None, (\n"
            "    f'tensors_log.grad_fn must be None — every element should have been detached first.'\n"
            ")\n"
            "assert tl.requires_grad is False, 'stacked detached tensors should not require grad'\n"
            "\n"
            "# Values: floats_log and tensors_log should agree elementwise.\n"
            "for i in range(5):\n"
            "    assert abs(fl[i] - tl[i].item()) < 1e-6, (\n"
            "        f'mismatch at step {i}: float={fl[i]}, tensor={tl[i].item()}. '\n"
            "        'Both logs should observe the SAME losses.'\n"
            "    )\n"
            "\n"
            "# Sanity: x**2 + 1 is always >= 1.\n"
            "for v in fl:\n"
            "    assert v >= 1.0, f'loss = x**2 + 1 should be >= 1, got {v}'\n"
            "\n"
            "# Larger run to confirm no graph blow-up.\n"
            "big = ex1_train_log(steps=200)\n"
            "assert len(big['floats_log']) == 200\n"
            "assert big['tensors_log'].shape == (200,)\n"
            "assert big['tensors_log'].grad_fn is None"
        ),
        "solution_body": (
            "def ex1_train_log(steps: int) -> dict:\n"
            "    x = t.randn(steps, requires_grad=False)\n"
            "    floats_log = []\n"
            "    tensors_log = []\n"
            "    for i in range(steps):\n"
            "        loss_i = (x[i] ** 2 + 1.0).requires_grad_(True)\n"
            "        floats_log.append(loss_i.item())              # Python float for logging\n"
            "        tensors_log.append(loss_i.detach().clone())   # detached tensor for buffering\n"
            "    stacked = t.stack(tensors_log)\n"
            "    return {\n"
            "        'floats_log': floats_log,\n"
            "        'tensors_log': stacked,\n"
            "        'sample_float_type': type(floats_log[0]).__name__,\n"
            "        'sample_tensor_type': type(tensors_log[0]).__name__,\n"
            "        'sample_tensor_grad_fn': tensors_log[0].grad_fn,\n"
            "    }"
        ),
        "solution_notes": (
            "**The graph-pinning bug.** Every operation on a "
            "`requires_grad=True` tensor records itself in the autograd "
            "graph for later `.backward()`. If you store `loss_i` directly "
            "(without `.detach()`), the graph that produced it stays "
            "rooted in your list — memory grows with `steps`, and "
            "subsequent `.backward()` calls re-traverse it. The "
            "`.detach()` call strips the `.grad_fn` reference, letting "
            "Python GC reclaim the graph nodes as soon as the next step's "
            "`loss_i` goes out of scope.\n\n"
            "**Why `.clone()` after `.detach()`.** `.detach()` returns a "
            "VIEW of the same storage with `requires_grad=False`. The "
            "`.clone()` makes a copy, which is important if the buffer "
            "outlives the source tensor — without the clone, mutations "
            "to the original (e.g. `loss_i.zero_()`) would corrupt every "
            "logged value. For 0-D scalars from `.item()`-flavored "
            "logging this is paranoia; for buffering intermediate "
            "activations it's essential.\n\n"
            "**The `.item()` synchronization cost.** On CUDA, `.item()` "
            "forces a host-device sync — it must wait for the kernel "
            "producing the value to finish before reading. One `.item()` "
            "per training step is fine (you're already waiting on the "
            "backward pass). One `.item()` per layer or per attention "
            "head tanks throughput — the kernel pipeline drains every "
            "call.\n\n"
            "**Why `float(loss)` is deprecated.** It worked for 0-D "
            "tensors via the `__float__` dunder, but PyTorch deprecated "
            "it for ambiguity — `float(some_1d_tensor)` would just take "
            "the first element. `.item()` raises a clear error on "
            "multi-element tensors, which is the safer behavior."
        ),
    },
]


# ===================================================================== verify

def verify_solutions(specs: list[dict]) -> None:
    """Exec each spec's solution_body against its test_body. Abort on any failure."""
    try:
        import torch as t
        import numpy as np
        from torch import Tensor
        import einops
        from einops import rearrange, reduce, repeat
    except ImportError as e:
        raise SystemExit(
            f"[build verify] missing runtime dep: {e}\n"
            f"  pip install torch numpy einops  # required for build-time solution verification\n"
            f"  refusing to write notebooks with unverified solutions."
        )

    failures: list[str] = []
    for spec in specs:
        atom = spec["atom_id"]
        ex_idx = spec["exercise_index"]
        label = f"{atom}/ex{ex_idx} ({spec['exercise_title']})"

        # Build a per-spec namespace with the standard imports.
        ns = {
            "t": t, "np": np, "Tensor": Tensor,
            "einops": einops,
            "rearrange": rearrange, "reduce": reduce, "repeat": repeat,
        }
        # Add any extra imports from the spec (e.g. matplotlib, math).
        for extra in spec.get("extra_imports", []) or []:
            try:
                exec(extra, ns)
            except Exception as e:
                failures.append(f"{label} — extra_import {extra!r} did not import: {e!r}")
                continue
        # Compile + exec the solution body.
        try:
            exec(spec["solution_body"], ns)
        except Exception as e:
            failures.append(f"{label} — solution_body did not compile: {e!r}")
            continue
        # Exec the test body inline. We DON'T indent it because we're not wrapping in a function;
        # we just want the asserts to run against the namespace built above.
        try:
            exec(spec["test_body"], ns)
        except Exception as e:
            failures.append(f"{label} — test_body assertion failed: {e!r}")
            continue

    if failures:
        print("[build verify] FAIL — refusing to emit notebooks.", file=sys.stderr)
        for line in failures:
            print(f"  X {line}", file=sys.stderr)
        raise SystemExit(1)

    print(f"[build verify] OK — {len(specs)} canonical solutions pass their tests.")


# ===================================================================== main

if __name__ == "__main__":
    verify_solutions(SPECS)
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"wrote {rel}")
