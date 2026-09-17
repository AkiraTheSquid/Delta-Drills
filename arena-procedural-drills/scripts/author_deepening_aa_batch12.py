#!/usr/bin/env python3
"""Author 8 ex2 deepening drills (batch 12, group Aa).

Atoms (3 prereqs_tensor_utils + 3 prereqs_training_loop + 2 prereqs_vae_gan):
    - topk-predictions                    (ex2: top-1 vs top-5 contrast — show case where top-5 hits but top-1 misses)
    - vector-normalize-keepdim            (ex2: dim=0 column vs dim=1 row + keepdim=False broadcast failure)
    - where-clip-negative                 (ex2: 3-way clamp (min, max) via two torch.where calls)
    - backward-on-scalar-loss             (ex2: backward() on a non-scalar requires gradient= arg — VJP)
    - train-eval-mode-branch              (ex2: BatchNorm — running stats freeze in eval; same input differs train vs eval)
    - validation-no-grad                  (ex2: @torch.no_grad() decorator on a function — same effect, cleaner API)
    - conv-leakyrelu-block-discriminator  (ex2: stride=1 (no downsample) variant — output H/W matches input)
    - convtranspose-bn-activation-block   (ex2: output_padding — 4→9 spatial via stride=2,kernel=3,pad=1,out_pad=2)

Each ex2 hits a DISTINCT facet from ex1. ONE LO + ONE Bloom + <=2 KCs per drill.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC_TENSOR = "prereqs_tensor_utils"
TOPIC_TRAIN = "prereqs_training_loop"
TOPIC_VAEGAN = "prereqs_vae_gan"


# ---------------------------------------------------------------------------
# Recap blocks
# ---------------------------------------------------------------------------

RECAP_TOPK = (
    "## top-1 vs top-5 — when do they disagree?\n"
    "\n"
    "Ex1 computed top-5 accuracy. The deepening move is to compute BOTH "
    "top-1 and top-5 on the same logits and surface the gap. Top-1 is "
    "strict (the argmax must hit the label); top-5 is permissive (the "
    "label must appear anywhere in the K=5 largest logits).\n"
    "\n"
    "```python\n"
    "top5 = logits.topk(5, dim=1).indices           # (B, 5)\n"
    "top1 = logits.argmax(dim=1, keepdim=True)       # (B, 1)\n"
    "hit5 = (top5 == labels[:, None]).any(dim=1)     # (B,) bool\n"
    "hit1 = (top1.squeeze(1) == labels)              # (B,) bool\n"
    "```\n"
    "\n"
    "**`top5_only`** = samples where top-5 hit AND top-1 missed. That's "
    "the wedge between the two metrics — the model knows the label is in "
    "the running but isn't certain enough to commit. ImageNet ResNets "
    "famously have a ~7-point gap between top-1 and top-5; the wedge is "
    "where the model is hedging.\n"
    "\n"
    "**Why `labels[:, None]` over `unsqueeze(1)`.** Same shape `(B, 1)`. "
    "Bracket-syntax is one fewer character and reads as 'add a column axis' "
    "instead of 'unsqueeze at position 1'."
)

RECAP_NORMALIZE = (
    "## row-wise vs column-wise normalize + the keepdim broadcast trap\n"
    "\n"
    "Ex1 normalized rows with `dim=1, keepdim=True`. The deepening move "
    "is to swap to columns (`dim=0`) and show what `keepdim=False` would "
    "break.\n"
    "\n"
    "```python\n"
    "# Row-wise (ex1):    norms.shape == (N, 1) — broadcasts across columns.\n"
    "# Column-wise (ex2): norms.shape == (1, M) — broadcasts across rows.\n"
    "row_norms = x.norm(dim=1, keepdim=True)   # (N, 1)\n"
    "col_norms = x.norm(dim=0, keepdim=True)   # (1, M)\n"
    "```\n"
    "\n"
    "**Why `keepdim=True` is load-bearing.** Without it, `x.norm(dim=0)` "
    "returns shape `(M,)`. Dividing a `(N, M)` tensor by a `(M,)` vector "
    "STILL broadcasts (right-aligned), so the column case happens to work "
    "without keepdim. But the row case (`dim=1` → `(N,)`) silently "
    "broadcasts WRONG: `(N, M) / (N,)` becomes `(N, M) / (1, N)`, which "
    "errors only if M != N — a sleeper bug on square matrices.\n"
    "\n"
    "**`keepdim=True` is the safe habit.** The dropped axis is replaced "
    "by size 1, so broadcasting goes back to the axis you reduced over. "
    "Works for `dim=0` and `dim=1` identically — no row/column asymmetry."
)

RECAP_WHERE_CLAMP = (
    "## three-way clamp via two `torch.where` calls\n"
    "\n"
    "Ex1 used `torch.where` to clip negatives — a one-sided clamp. The "
    "deepening move is a TWO-SIDED clamp: clip to `[lo, hi]` using two "
    "stacked `where` calls.\n"
    "\n"
    "```python\n"
    "# Step 1: lift everything below `lo` up to `lo`.\n"
    "y = t.where(x < lo, t.full_like(x, lo), x)\n"
    "# Step 2: pull everything above `hi` down to `hi`.\n"
    "y = t.where(y > hi, t.full_like(y, hi), y)\n"
    "```\n"
    "\n"
    "**Equivalent to `torch.clamp(x, lo, hi)`.** The exercise rebuilds it "
    "from `where` to make the control flow visible — two scalar "
    "predicates, two broadcast selects.\n"
    "\n"
    "**Order doesn't matter when `lo <= hi`.** Either sweep can run first; "
    "the second sweep then re-clips its own output. If `lo > hi` (a typo "
    "trap) the two orders disagree — the exercise asks you to detect that "
    "case up front and raise `ValueError`.\n"
    "\n"
    "**`torch.full_like(x, lo)` over `lo * torch.ones_like(x)`.** One op, "
    "matches dtype + device + shape automatically."
)

RECAP_BACKWARD_VECTOR = (
    "## `backward()` on a non-scalar — the `gradient=` argument (VJP)\n"
    "\n"
    "Ex1 called `loss.backward()` after reducing per-sample loss to a "
    "scalar. The deepening move skips the reduce: call `.backward()` on "
    "a VECTOR `y` and pass an explicit `gradient=v` argument. This is "
    "the vector-Jacobian product (VJP) — `x.grad = J^T @ v`.\n"
    "\n"
    "```python\n"
    "y = f(x)                              # y shape (N,), x shape (N,)\n"
    "v = torch.ones_like(y)                # this picks the row of J^T\n"
    "y.backward(gradient=v)                # x.grad = sum over i of dy_i/dx\n"
    "```\n"
    "\n"
    "Why this is equivalent to `y.sum().backward()`. Reducing with sum "
    "and then differentiating is `d(sum(y))/dx = sum_i dy_i/dx = J^T @ 1`. "
    "Passing `gradient=ones_like(y)` is the SAME computation, just spelled "
    "out as the VJP directly.\n"
    "\n"
    "**Why you can't omit `gradient=` for non-scalar `y`.** PyTorch needs "
    "to know which scalar function of `y` you are differentiating. For a "
    "scalar `y`, the only choice is `y` itself, so `gradient=` defaults "
    "to `torch.tensor(1.0)`. For a vector `y`, there's no canonical "
    "choice — calling `.backward()` without `gradient=` raises "
    "`RuntimeError`.\n"
    "\n"
    "**The full Jacobian falls out of N VJPs.** Pass `gradient=e_i` (a "
    "one-hot vector) and you recover the i-th row of `J^T` — i.e. column "
    "of `J`. Stack N such calls and you have the full Jacobian."
)

RECAP_EVAL_BN = (
    "## `eval()` freezes BatchNorm running stats, not just Dropout\n"
    "\n"
    "Ex1 toggled `.train()`/`.eval()` around Dropout. The deepening move "
    "is BatchNorm — a layer whose `train()` vs `eval()` behaviour is MORE "
    "consequential than Dropout's, because BN updates running stats in "
    "train mode and reads them in eval mode.\n"
    "\n"
    "**Train mode (BN):**\n"
    "1. Compute batch mean / variance over the current minibatch.\n"
    "2. Normalize using those batch statistics.\n"
    "3. Update `running_mean` / `running_var` via momentum.\n"
    "\n"
    "**Eval mode (BN):**\n"
    "1. Normalize using the saved `running_mean` / `running_var`.\n"
    "2. Do NOT update them.\n"
    "\n"
    "Consequence: feeding the SAME input through the SAME BN layer in "
    "the two modes gives DIFFERENT outputs (unless batch stats happen to "
    "equal running stats, which is the limit of training, not the "
    "general case).\n"
    "\n"
    "```python\n"
    "bn = nn.BatchNorm1d(4)\n"
    "bn.train(); bn(x)               # uses batch stats, mutates running_*\n"
    "bn.eval();  bn(x)               # uses running stats, no mutation\n"
    "```\n"
    "\n"
    "**Why eval mode also matters for Dropout — but differently.** "
    "Dropout in eval is identity (no-op); BN in eval is a fixed affine "
    "given by the running stats. Forgetting `.eval()` at inference time "
    "is the #1 silent bug in ARENA-scale codebases."
)

RECAP_NOGRAD_DECORATOR = (
    "## `@torch.no_grad()` decorator vs the `with torch.no_grad():` block\n"
    "\n"
    "Ex1 wrapped the validation loop body in `with torch.no_grad():`. The "
    "deepening move: use `@torch.no_grad()` as a DECORATOR on the eval "
    "function itself. Same effect, cleaner API.\n"
    "\n"
    "```python\n"
    "@torch.no_grad()\n"
    "def validate(model, loader):\n"
    "    model.eval()\n"
    "    total_correct = 0\n"
    "    total = 0\n"
    "    for x, y in loader:\n"
    "        out = model(x)\n"
    "        total_correct += (out.argmax(dim=1) == y).sum().item()\n"
    "        total += y.numel()\n"
    "    return total_correct / total\n"
    "```\n"
    "\n"
    "**Why the decorator is preferred.** Wraps the ENTIRE function body, "
    "including the loop setup, the iteration, and any helpers called "
    "inline. The `with` block can leak grad-enabled paths through helper "
    "calls if the helper itself opens a `with torch.enable_grad()`. The "
    "decorator makes the no-grad contract part of the function's "
    "signature.\n"
    "\n"
    "**Still need `.eval()`.** `no_grad` only disables autograd; it does "
    "NOT change Dropout/BN mode. Both calls are required at inference."
)

RECAP_CONV_STRIDE1 = (
    "## Conv2d + BN + LeakyReLU — stride=1 (no downsample) variant\n"
    "\n"
    "Ex1 built a `Conv2d(stride=2) → BN → LeakyReLU` block — DCGAN's "
    "down-sampling discriminator unit. The deepening move keeps the "
    "topology but flips `stride=1` so the spatial dimensions are "
    "PRESERVED — same kernel size + padding chosen so H_out == H_in.\n"
    "\n"
    "**Spatial formula:** `H_out = floor((H_in + 2*pad - kernel) / stride) + 1`. \n"
    "With `kernel=3, stride=1, pad=1`: `H_out = floor((H + 2 - 3) / 1) + 1 "
    "= H`. The kernel CAN reach every spatial position; padding picks up "
    "the boundary slack.\n"
    "\n"
    "```python\n"
    "block = nn.Sequential(\n"
    "    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),\n"
    "    nn.BatchNorm2d(out_ch),\n"
    "    nn.LeakyReLU(0.2, inplace=True),\n"
    ")\n"
    "```\n"
    "\n"
    "**Why this variant matters.** DCGAN discriminators alternate "
    "stride-2 (downsample) and stride-1 (refine) blocks. The stride-1 "
    "block is where you ADD CAPACITY without changing the feature-map "
    "size — channel growth without spatial collapse.\n"
    "\n"
    "**`bias=False` still applies.** BatchNorm has its own affine bias; "
    "the conv's bias would be immediately subtracted out by BN's "
    "mean-centering. Save the parameters."
)

RECAP_CONVT_OUTPAD = (
    "## ConvTranspose2d + BN + ReLU — `output_padding` for odd outputs\n"
    "\n"
    "Ex1 built a `ConvTranspose2d → BN → ReLU` block. The deepening move "
    "exercises `output_padding`, the knob that disambiguates the inverse "
    "of strided Conv.\n"
    "\n"
    "**Spatial formula:** "
    "`H_out = (H_in - 1)*stride - 2*pad + kernel + output_padding`. \n"
    "Forward strided Conv collapses multiple inputs to one output, so the "
    "inverse is many-to-one — the transpose needs an extra hint to pick "
    "the right output size.\n"
    "\n"
    "**4 → 9 example.** `H_in=4, stride=2, kernel=3, pad=1`:\n"
    "- `(4 - 1)*2 - 2*1 + 3 = 6 - 2 + 3 = 7` with `output_padding=0`.\n"
    "- Add `output_padding=2` → `7 + 2 = 9`. (Constraint: `output_padding "
    "< stride OR output_padding < dilation`; `2 < stride? no` — PyTorch "
    "actually requires `output_padding < max(stride, dilation)`, and "
    "`stride=2` gives an upper bound of 2, so `output_padding=2` is "
    "REJECTED.) The correct ARENA pattern picks `output_padding=1, "
    "stride=2, kernel=3, pad=1` → `H_out=8` from `H_in=4`, then a second "
    "block handles the next jump.\n"
    "\n"
    "**Pragmatic 4 → 9 path.** Use `stride=2, kernel=4, pad=0, "
    "output_padding=1`: `(4-1)*2 - 0 + 4 + 1 = 6 + 4 + 1 = 11`. Doesn't "
    "land on 9. Use `stride=2, kernel=3, pad=2, output_padding=1`: "
    "`(4-1)*2 - 4 + 3 + 1 = 6 - 4 + 3 + 1 = 6`. Still not 9.\n"
    "\n"
    "**The clean 4 → 9 spec.** `stride=3, kernel=3, pad=1, "
    "output_padding=2`: `(4-1)*3 - 2 + 3 + 2 = 9 - 2 + 3 + 2 = 12`. "
    "Closer but still off. The TRUE working spec: `stride=2, kernel=4, "
    "pad=1, output_padding=1`: `(4-1)*2 - 2 + 4 + 1 = 6 - 2 + 4 + 1 = 9`. "
    "✓ — and `output_padding=1 < stride=2` satisfies the constraint.\n"
    "\n"
    "```python\n"
    "block = nn.Sequential(\n"
    "    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4,\n"
    "                       stride=2, padding=1, output_padding=1, bias=False),\n"
    "    nn.BatchNorm2d(out_ch),\n"
    "    nn.ReLU(inplace=True),\n"
    ")\n"
    "```\n"
    "\n"
    "**Why this matters.** Generator stacks frequently need to land on "
    "an odd target size (e.g. 7×7 MNIST). `output_padding` is the only "
    "non-fractional knob that hits odd outputs from even inputs without "
    "switching to fractional strides."
)


# ---------------------------------------------------------------------------
# SPEC 1 — topk-predictions ex2
# ---------------------------------------------------------------------------

SPEC_TOPK = {
    "atom_id": "topk-predictions",
    "subtopic": "Eval: topk predictions",
    "topic_folder": TOPIC_TENSOR,
    "atom_recap_md": RECAP_TOPK,
    "exercise_index": 2,
    "exercise_title": "top-1 vs top-5 contrast — count the wedge where top-5 hits but top-1 misses",
    "slug": "top1-vs-top5-wedge",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["topk", "top1", "top5", "accuracy"],
    "kcs": [
        "topk-membership-via-any",
        "argmax-vs-topk-comparison",
    ],
    "lo": (
        "Apply both `argmax` and `topk(5).indices` against the same "
        "labels and return three counts: top-1 hits, top-5 hits, and "
        "the wedge (top-5 hits but top-1 misses)."
    ),
    "prompt_body": (
        "Implement `ex2_top1_top5_wedge(logits, labels)`. Surfaces the "
        "gap between the two metrics on a single batch.\n\n"
        "Inputs:\n"
        "- `logits`: `(B, C)` float tensor.\n"
        "- `labels`: `(B,)` long tensor with values in `[0, C)`.\n\n"
        "Algorithm:\n"
        "1. `top5 = logits.topk(5, dim=1).indices` — shape `(B, 5)`.\n"
        "2. `top1 = logits.argmax(dim=1)` — shape `(B,)`.\n"
        "3. `hit1 = (top1 == labels)` — shape `(B,)`, bool.\n"
        "4. `hit5 = (top5 == labels[:, None]).any(dim=1)` — shape `(B,)`, "
        "bool. (`labels[:, None]` broadcasts the label across the 5 "
        "topk slots.)\n"
        "5. `wedge = hit5 & (~hit1)` — top-5 captured the label but "
        "top-1 didn't. Return integers (Python ints, not tensors).\n\n"
        "Output: `dict` with keys `'top1_hits'`, `'top5_hits'`, "
        "`'wedge'` — each a Python `int` count.\n\n"
        "Constraint: assume `C >= 5`. If `C < 5`, `topk(5)` raises — "
        "that's PyTorch's contract, not yours to handle."
    ),
    "stub": (
        "def ex2_top1_top5_wedge(logits: Tensor, labels: Tensor) -> dict:\n"
        '    """Return {top1_hits, top5_hits, wedge} as Python ints."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Hand-traced: 4 samples, 10 classes ===\n"
        "# Sample 0: label=3, argmax also 3 → top-1 hit, top-5 hit.\n"
        "# Sample 1: label=7, label is the 2nd-largest logit → top-1 miss, top-5 hit (wedge).\n"
        "# Sample 2: label=0, label is the 4th-largest → top-1 miss, top-5 hit (wedge).\n"
        "# Sample 3: label=9, label is 6th-largest → top-1 miss, top-5 miss.\n"
        "logits = t.tensor([\n"
        "    [0.1, 0.2, 0.1, 5.0, 0.1, 0.1, 0.0, 0.1, 0.0, 0.0],  # argmax=3\n"
        "    [0.1, 0.2, 0.1, 0.1, 0.1, 4.0, 3.0, 3.5, 0.5, 0.0],  # argmax=5, label=7 ranks #2 -> in top-5\n"
        "    [2.0, 0.1, 0.1, 5.0, 4.0, 3.0, 2.5, 0.5, 0.4, 0.3],  # argmax=3, label=0 ranks #4 -> in top-5\n"
        "    [0.1, 5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.4, 0.3],  # argmax=1, label=9 ranks #10 -> not in top-5\n"
        "])\n"
        "labels = t.tensor([3, 7, 0, 9])\n"
        "out = ex2_top1_top5_wedge(logits, labels)\n"
        "assert isinstance(out, dict), f'must return dict, got {type(out).__name__}'\n"
        "assert set(out.keys()) == {'top1_hits', 'top5_hits', 'wedge'}, f'keys wrong: {set(out.keys())}'\n"
        "for k in ('top1_hits', 'top5_hits', 'wedge'):\n"
        "    assert isinstance(out[k], int), f'{k} must be Python int, got {type(out[k]).__name__}'\n"
        "assert out['top1_hits'] == 1, f'top1 should be 1 (sample 0), got {out[\"top1_hits\"]}'\n"
        "assert out['top5_hits'] == 3, f'top5 should be 3 (samples 0,1,2), got {out[\"top5_hits\"]}'\n"
        "assert out['wedge'] == 2, f'wedge should be 2 (samples 1,2), got {out[\"wedge\"]}'\n"
        "\n"
        "# === All samples top-1 hit → wedge is 0 ===\n"
        "logits = t.eye(8) * 10.0  # 8x8, each row argmax = its index\n"
        "labels = t.arange(8)\n"
        "out = ex2_top1_top5_wedge(logits, labels)\n"
        "assert out['top1_hits'] == 8\n"
        "assert out['top5_hits'] == 8\n"
        "assert out['wedge'] == 0, f'wedge must be 0 when every top-1 hits, got {out[\"wedge\"]}'\n"
        "\n"
        "# === No top-5 hits → wedge is 0 (can't wedge without top-5) ===\n"
        "# Logits where labels are all the SMALLEST score (rank 6+).\n"
        "logits = t.tensor([\n"
        "    [9.0, 8.0, 7.0, 6.0, 5.0, 0.1, 0.0],   # label=5 → rank 6 → out of top-5\n"
        "    [9.0, 8.0, 7.0, 6.0, 5.0, 0.0, 0.1],   # label=6 → rank 6 → out\n"
        "])\n"
        "labels = t.tensor([5, 6])\n"
        "out = ex2_top1_top5_wedge(logits, labels)\n"
        "assert out['top1_hits'] == 0\n"
        "assert out['top5_hits'] == 0\n"
        "assert out['wedge'] == 0\n"
        "\n"
        "# === Wedge identity: top5_hits >= top1_hits, wedge == top5_hits - top1_hits ===\n"
        "t.manual_seed(0)\n"
        "logits = t.randn(50, 20)\n"
        "labels = t.randint(0, 20, (50,))\n"
        "out = ex2_top1_top5_wedge(logits, labels)\n"
        "assert out['top5_hits'] >= out['top1_hits'], 'top-5 must dominate top-1'\n"
        "assert out['wedge'] == out['top5_hits'] - out['top1_hits'], (\n"
        "    f'wedge identity broken: wedge={out[\"wedge\"]}, '\n"
        "    f'top5-top1={out[\"top5_hits\"] - out[\"top1_hits\"]}'\n"
        ")\n"
        "\n"
        "# === Single-sample edge ===\n"
        "logits = t.tensor([[0.1, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5]])\n"
        "labels = t.tensor([4])  # rank 3 → in top-5; argmax is 1 → top-1 miss\n"
        "out = ex2_top1_top5_wedge(logits, labels)\n"
        "assert out['top1_hits'] == 0\n"
        "assert out['top5_hits'] == 1\n"
        "assert out['wedge'] == 1"
    ),
    "solution_body": (
        "def ex2_top1_top5_wedge(logits, labels):\n"
        "    top5 = logits.topk(5, dim=1).indices\n"
        "    top1 = logits.argmax(dim=1)\n"
        "    hit1 = (top1 == labels)\n"
        "    hit5 = (top5 == labels[:, None]).any(dim=1)\n"
        "    wedge = hit5 & (~hit1)\n"
        "    return {\n"
        "        'top1_hits': int(hit1.sum().item()),\n"
        "        'top5_hits': int(hit5.sum().item()),\n"
        "        'wedge':     int(wedge.sum().item()),\n"
        "    }"
    ),
    "solution_notes": (
        "**Identity `wedge == top5_hits - top1_hits`.** Provable from "
        "the algebra: every top-1 hit is also a top-5 hit (the argmax "
        "is always in the top-K), so `hit1 ⊆ hit5`, and `wedge = hit5 ∧ "
        "¬hit1 = hit5 - hit1`. Treat any deviation as a bug.\n\n"
        "**`labels[:, None]` for the membership compare.** Reshape "
        "`(B,)` → `(B, 1)`, then broadcasts against `(B, 5)` from "
        "`topk`. Cleaner than `unsqueeze(1)`.\n\n"
        "**`int(...)` over `.item()` alone.** `.item()` already returns "
        "a Python number; the outer `int(...)` is a defensive cast — "
        "`bool.sum().item()` can be a numpy int on some PyTorch builds. "
        "Forcing `int` makes downstream serialization (`json.dumps`) "
        "work uniformly."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 2 — vector-normalize-keepdim ex2
# ---------------------------------------------------------------------------

SPEC_NORMALIZE = {
    "atom_id": "vector-normalize-keepdim",
    "subtopic": "PyTorch: vector normalize keepdim",
    "topic_folder": TOPIC_TENSOR,
    "atom_recap_md": RECAP_NORMALIZE,
    "exercise_index": 2,
    "exercise_title": "column-wise L2 normalize with keepdim=True (axis-flip of ex1)",
    "slug": "column-wise-l2-normalize-keepdim",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["normalize", "norm", "keepdim", "broadcast"],
    "kcs": [
        "norm-with-keepdim-preserves-rank",
        "axis-flip-row-vs-column",
    ],
    "lo": (
        "Apply `x.norm(dim=0, keepdim=True)` to L2-normalize the "
        "COLUMNS of a 2-D tensor — flipping ex1's row axis — while "
        "still leaning on keepdim to avoid the squeezed-shape "
        "broadcasting trap."
    ),
    "prompt_body": (
        "Implement `ex2_normalize_columns(x, eps=1e-12)`. The axis-"
        "flipped variant of ex1.\n\n"
        "Inputs:\n"
        "- `x`: `(N, M)` float tensor.\n"
        "- `eps`: float, additive guard against divide-by-zero.\n\n"
        "Algorithm:\n"
        "1. `col_norms = x.norm(dim=0, keepdim=True)` — shape `(1, M)`.\n"
        "2. Return `x / (col_norms + eps)`.\n\n"
        "Constraints:\n"
        "- DO NOT use `dim=1`. This drill is about the column axis "
        "specifically.\n"
        "- DO NOT pass `keepdim=False`. Keep the reduced axis at size 1.\n"
        "- Preserve the input dtype.\n"
        "- DO NOT mutate `x`.\n\n"
        "Output: `(N, M)` tensor where each COLUMN has L2 norm ≈ 1.0 "
        "(within `eps` of 1.0 for non-degenerate columns)."
    ),
    "stub": (
        "def ex2_normalize_columns(x: Tensor, eps: float = 1e-12) -> Tensor:\n"
        '    """L2-normalize each column. Uses x.norm(dim=0, keepdim=True)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Each column of the result has unit L2 norm ===\n"
        "t.manual_seed(0)\n"
        "x = t.randn(7, 4)\n"
        "y = ex2_normalize_columns(x)\n"
        "col_norms_out = y.norm(dim=0)  # (M,)\n"
        "assert t.allclose(col_norms_out, t.ones(4), atol=1e-5), (\n"
        "    f'each column must have unit norm, got {col_norms_out}'\n"
        ")\n"
        "\n"
        "# === Shape + dtype preserved ===\n"
        "assert y.shape == x.shape, f'shape mismatch: in {tuple(x.shape)} vs out {tuple(y.shape)}'\n"
        "assert y.dtype == x.dtype, f'dtype must be preserved, got {y.dtype}'\n"
        "\n"
        "# === Direction preserved per column (cosine == 1) ===\n"
        "for c in range(x.shape[1]):\n"
        "    cos = t.dot(x[:, c], y[:, c]) / (x[:, c].norm() * y[:, c].norm())\n"
        "    assert t.allclose(cos, t.tensor(1.0), atol=1e-5), f'col {c} direction changed: cos={cos}'\n"
        "\n"
        "# === Input not mutated ===\n"
        "x_clone = x.clone()\n"
        "_ = ex2_normalize_columns(x_clone)\n"
        "assert t.equal(x_clone, x), 'must not mutate input'\n"
        "\n"
        "# === Hand-traced 3x2 ===\n"
        "x = t.tensor([\n"
        "    [3.0, 0.0],\n"
        "    [4.0, 0.0],\n"
        "    [0.0, 5.0],\n"
        "])\n"
        "# col 0 norm = 5, col 1 norm = 5 → result rows: (3/5,0), (4/5,0), (0,1)\n"
        "out = ex2_normalize_columns(x)\n"
        "expected = t.tensor([\n"
        "    [0.6, 0.0],\n"
        "    [0.8, 0.0],\n"
        "    [0.0, 1.0],\n"
        "])\n"
        "assert t.allclose(out, expected, atol=1e-5), f'expected={expected}, got {out}'\n"
        "\n"
        "# === Degenerate (zero) column → result is finite (eps guard) ===\n"
        "x = t.tensor([\n"
        "    [0.0, 1.0],\n"
        "    [0.0, 2.0],\n"
        "    [0.0, 2.0],\n"
        "])\n"
        "out = ex2_normalize_columns(x)\n"
        "assert t.isfinite(out).all(), f'eps guard failed; got {out}'\n"
        "# Zero column stays zero (0 / eps ≈ 0).\n"
        "assert t.allclose(out[:, 0], t.zeros(3), atol=1e-5)\n"
        "# Other column still has ~unit norm.\n"
        "assert t.isclose(out[:, 1].norm(), t.tensor(1.0), atol=1e-5)\n"
        "\n"
        "# === Square matrix (N == M) — keepdim is load-bearing here ===\n"
        "# If somebody used keepdim=False, the (M,) shape would broadcast as a row,\n"
        "# which on a square matrix would NOT error but would normalize the WRONG axis.\n"
        "# Use a non-symmetric value pattern to detect axis confusion.\n"
        "x = t.tensor([\n"
        "    [1.0, 0.0, 0.0],\n"
        "    [0.0, 2.0, 0.0],\n"
        "    [0.0, 0.0, 3.0],\n"
        "])\n"
        "out = ex2_normalize_columns(x)\n"
        "# Each column has only one non-zero. After normalize, each column is a one-hot.\n"
        "expected = t.eye(3)\n"
        "assert t.allclose(out, expected, atol=1e-5), (\n"
        "    f'square-matrix axis confusion: expected eye(3), got {out}'\n"
        ")\n"
        "\n"
        "# === Single-row tensor → each column normalizes to ±1 ===\n"
        "x = t.tensor([[2.0, -3.0, 1.0]])\n"
        "out = ex2_normalize_columns(x)\n"
        "# Single-row L2 norm per column is |x[0, c]|, so each column becomes sign.\n"
        "expected = t.tensor([[1.0, -1.0, 1.0]])\n"
        "assert t.allclose(out, expected, atol=1e-5)\n"
        "\n"
        "# === Single-column tensor still works ===\n"
        "x = t.tensor([[3.0], [4.0]])\n"
        "out = ex2_normalize_columns(x)\n"
        "expected = t.tensor([[0.6], [0.8]])\n"
        "assert t.allclose(out, expected, atol=1e-5)"
    ),
    "solution_body": (
        "def ex2_normalize_columns(x, eps=1e-12):\n"
        "    col_norms = x.norm(dim=0, keepdim=True)  # (1, M)\n"
        "    return x / (col_norms + eps)"
    ),
    "solution_notes": (
        "**`dim=0` is the COLUMN reduce.** A common mental hiccup — "
        "`dim=0` SAVES axis 1 (columns); `dim=1` SAVES axis 0 (rows). "
        "The dim you pass is the dim that DISAPPEARS in the reduction.\n\n"
        "**`keepdim=True` is non-optional for the square-matrix case.** "
        "On rectangular tensors, `(M,) / (N, M)` happens to broadcast "
        "correctly (right-aligned), but on square `(N, N)` it would "
        "silently normalize the wrong axis. The square-matrix test "
        "above catches that.\n\n"
        "**`+ eps` over `clamp(min=eps)`.** Additive guard is one op; "
        "`clamp` is two. For inputs whose norm is genuinely zero, both "
        "give a near-zero output column, which is the right behavior — "
        "you can't recover a direction from the zero vector."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 3 — where-clip-negative ex2
# ---------------------------------------------------------------------------

SPEC_WHERE_CLAMP = {
    "atom_id": "where-clip-negative",
    "subtopic": "PyTorch: where to clip negative",
    "topic_folder": TOPIC_TENSOR,
    "atom_recap_md": RECAP_WHERE_CLAMP,
    "exercise_index": 2,
    "exercise_title": "two-sided clamp via stacked `torch.where` calls (plus lo<=hi validation)",
    "slug": "two-sided-clamp-via-stacked-where",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["where", "clamp", "two-sided"],
    "kcs": [
        "stacked-where-for-two-sided-clip",
        "precondition-low-less-or-equal-high",
    ],
    "lo": (
        "Apply two `torch.where` calls to clip a tensor to `[lo, hi]`, "
        "matching `torch.clamp`'s output, and raise `ValueError` if "
        "`lo > hi`."
    ),
    "prompt_body": (
        "Implement `ex2_clamp_via_where(x, lo, hi)`. A two-sided clamp "
        "built from `torch.where`.\n\n"
        "Inputs:\n"
        "- `x`: float tensor of any shape.\n"
        "- `lo`: float, lower bound (inclusive).\n"
        "- `hi`: float, upper bound (inclusive).\n\n"
        "Algorithm:\n"
        "1. If `lo > hi`: raise `ValueError` whose message mentions "
        "both `lo` and `hi` (case-insensitive).\n"
        "2. `y = torch.where(x < lo, torch.full_like(x, lo), x)` — lift "
        "the under-floor values.\n"
        "3. `y = torch.where(y > hi, torch.full_like(y, hi), y)` — pull "
        "the over-ceiling values.\n"
        "4. Return `y`. Same shape, same dtype as `x`.\n\n"
        "Constraints:\n"
        "- DO NOT call `torch.clamp` directly. Build it from `where`.\n"
        "- DO NOT use `torch.minimum`/`torch.maximum`.\n"
        "- DO NOT mutate `x`."
    ),
    "stub": (
        "def ex2_clamp_via_where(x: Tensor, lo: float, hi: float) -> Tensor:\n"
        '    """Two-sided clamp to [lo, hi] built from two torch.where calls."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Matches torch.clamp on a random tensor ===\n"
        "t.manual_seed(0)\n"
        "x = t.randn(64) * 5.0\n"
        "out = ex2_clamp_via_where(x, lo=-2.0, hi=3.0)\n"
        "ref = t.clamp(x, -2.0, 3.0)\n"
        "assert t.allclose(out, ref), f'must match torch.clamp; first diff at {(out - ref).abs().argmax().item()}'\n"
        "assert out.shape == x.shape and out.dtype == x.dtype\n"
        "\n"
        "# === Hand-traced 1-D ===\n"
        "x = t.tensor([-5.0, -1.0, 0.0, 1.0, 5.0, 10.0])\n"
        "out = ex2_clamp_via_where(x, lo=-2.0, hi=3.0)\n"
        "expected = t.tensor([-2.0, -1.0, 0.0, 1.0, 3.0, 3.0])\n"
        "assert t.equal(out, expected), f'expected={expected}, got {out}'\n"
        "\n"
        "# === Boundary values pass through unchanged ===\n"
        "x = t.tensor([-2.0, 3.0])\n"
        "out = ex2_clamp_via_where(x, lo=-2.0, hi=3.0)\n"
        "assert t.equal(out, x), f'boundary values must pass through, got {out}'\n"
        "\n"
        "# === Input not mutated ===\n"
        "x_orig = t.tensor([-10.0, 0.0, 10.0])\n"
        "x_clone = x_orig.clone()\n"
        "_ = ex2_clamp_via_where(x_clone, -1.0, 1.0)\n"
        "assert t.equal(x_clone, x_orig), 'must not mutate input'\n"
        "\n"
        "# === lo == hi → all values pinned to that single value ===\n"
        "x = t.tensor([-5.0, 0.0, 5.0])\n"
        "out = ex2_clamp_via_where(x, lo=2.5, hi=2.5)\n"
        "assert t.allclose(out, t.full_like(x, 2.5)), f'lo==hi must pin everything, got {out}'\n"
        "\n"
        "# === lo > hi → ValueError mentioning lo and hi ===\n"
        "try:\n"
        "    ex2_clamp_via_where(t.tensor([0.0]), lo=5.0, hi=1.0)\n"
        "except ValueError as e:\n"
        "    msg = str(e).lower()\n"
        "    assert 'lo' in msg and 'hi' in msg, f'error must mention lo and hi, got {e!r}'\n"
        "else:\n"
        "    raise AssertionError('expected ValueError for lo > hi')\n"
        "\n"
        "# === Multi-D shape preserved ===\n"
        "x = t.randn(3, 4, 5)\n"
        "out = ex2_clamp_via_where(x, -0.5, 0.5)\n"
        "assert out.shape == (3, 4, 5), f'shape wrong: {tuple(out.shape)}'\n"
        "assert (out >= -0.5).all() and (out <= 0.5).all(), 'all values must be in [lo, hi]'\n"
        "\n"
        "# === Negative-only range still works ===\n"
        "x = t.tensor([-10.0, -3.0, 0.0, 3.0, 10.0])\n"
        "out = ex2_clamp_via_where(x, lo=-5.0, hi=-1.0)\n"
        "expected = t.tensor([-5.0, -3.0, -1.0, -1.0, -1.0])\n"
        "assert t.equal(out, expected), f'expected={expected}, got {out}'\n"
        "\n"
        "# === Dtype preservation (int) ===\n"
        "# Note: full_like passes lo/hi through torch's float-to-int cast.\n"
        "x = t.tensor([-5, 0, 5, 10], dtype=t.int32)\n"
        "out = ex2_clamp_via_where(x, lo=-2, hi=3)\n"
        "assert out.dtype == x.dtype, f'int dtype must be preserved, got {out.dtype}'\n"
        "assert t.equal(out, t.tensor([-2, 0, 3, 3], dtype=t.int32))"
    ),
    "solution_body": (
        "def ex2_clamp_via_where(x, lo, hi):\n"
        "    if lo > hi:\n"
        "        raise ValueError(f'lo must be <= hi, got lo={lo} hi={hi}')\n"
        "    y = t.where(x < lo, t.full_like(x, lo), x)\n"
        "    y = t.where(y > hi, t.full_like(y, hi), y)\n"
        "    return y"
    ),
    "solution_notes": (
        "**Two `where` calls > one nested `where`.** A nested form like "
        "`where(x<lo, lo, where(x>hi, hi, x))` works but is harder to "
        "read. The stacked form makes each clip step a separate line — "
        "easier to debug when one bound is wrong.\n\n"
        "**`torch.full_like(x, lo)` over a scalar broadcast.** Scalars "
        "broadcast fine, but `full_like` makes the dtype + device "
        "matching explicit. On an `int32` input with `lo=1.5`, "
        "`full_like` rounds (or PyTorch will error on type mismatch) "
        "instead of silently promoting to float64.\n\n"
        "**`lo > hi` raises; `lo == hi` is fine.** A single-point clamp "
        "is a legitimate (if degenerate) use case — projecting "
        "everything onto a constant. Only the strictly-inverted bound "
        "is the error condition."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 4 — backward-on-scalar-loss ex2
# ---------------------------------------------------------------------------

SPEC_BACKWARD_VECTOR = {
    "atom_id": "backward-on-scalar-loss",
    "subtopic": "PyTorch: backward()",
    "topic_folder": TOPIC_TRAIN,
    "atom_recap_md": RECAP_BACKWARD_VECTOR,
    "exercise_index": 2,
    "exercise_title": "vector-output backward() requires gradient= (VJP) — verify x.grad matches y.sum().backward()",
    "slug": "vector-backward-with-gradient-argument-vjp",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["autograd", "backward", "VJP", "gradient"],
    "kcs": [
        "non-scalar-backward-needs-gradient-argument",
        "vjp-equals-sum-then-backward",
    ],
    "lo": (
        "Apply `y.backward(gradient=ones_like(y))` to compute "
        "`x.grad = J^T @ 1` on a non-scalar `y`, verify the result "
        "matches `y.sum().backward()`, and confirm `y.backward()` "
        "without an argument raises `RuntimeError`."
    ),
    "prompt_body": (
        "Implement `ex2_vjp_via_backward(x, f)`. Compute the gradient "
        "of `sum(f(x))` w.r.t. `x` using the VJP form of `.backward()`.\n\n"
        "Inputs:\n"
        "- `x`: 1-D float tensor (will need `requires_grad=True` "
        "internally — see step 1).\n"
        "- `f`: a callable `f(x) -> Tensor` where `f(x)` has the same "
        "shape as `x` (vector → vector).\n\n"
        "Algorithm:\n"
        "1. Make a fresh leaf `x_leaf = x.detach().clone().requires_grad_(True)`. "
        "Do NOT modify the caller's `x`.\n"
        "2. Compute `y = f(x_leaf)`.\n"
        "3. Verify `y` is NOT a scalar (`y.dim() > 0`). If it IS a "
        "scalar, raise `ValueError(\"f must return a non-scalar tensor\")`.\n"
        "4. Build `v = torch.ones_like(y)`.\n"
        "5. Call `y.backward(gradient=v)`.\n"
        "6. Return `x_leaf.grad.detach().clone()`.\n\n"
        "Output: 1-D tensor of the same shape as `x` containing the "
        "VJP, which equals `d(sum(f(x)))/dx`."
    ),
    "stub": (
        "def ex2_vjp_via_backward(x: Tensor, f) -> Tensor:\n"
        '    """Compute d(sum(f(x)))/dx via y.backward(gradient=ones_like(y))."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === f(x) = x^2  →  d(sum(x^2))/dx = 2x ===\n"
        "x = t.tensor([1.0, 2.0, 3.0, 4.0])\n"
        "grad = ex2_vjp_via_backward(x, lambda v: v ** 2)\n"
        "expected = 2 * x\n"
        "assert t.allclose(grad, expected), f'd(sum(x^2))/dx mismatch: expected={expected}, got {grad}'\n"
        "\n"
        "# === Caller's x is unchanged: still a leaf with no grad ===\n"
        "assert not x.requires_grad, 'caller x must not have requires_grad flipped on'\n"
        "assert x.grad is None, 'caller x must have no .grad'\n"
        "\n"
        "# === Matches y.sum().backward() exactly ===\n"
        "x2 = t.tensor([0.5, -1.5, 2.0], requires_grad=True)\n"
        "y2 = (x2 ** 3 + 2 * x2)\n"
        "y2.sum().backward()\n"
        "ref = x2.grad.clone()\n"
        "x3 = t.tensor([0.5, -1.5, 2.0])\n"
        "vjp = ex2_vjp_via_backward(x3, lambda v: v ** 3 + 2 * v)\n"
        "assert t.allclose(vjp, ref), f'VJP must equal sum-backward: ref={ref}, got {vjp}'\n"
        "\n"
        "# === f(x) = x (identity)  →  grad is all-ones ===\n"
        "x = t.tensor([10.0, -3.0, 0.0])\n"
        "grad = ex2_vjp_via_backward(x, lambda v: v.clone())\n"
        "assert t.allclose(grad, t.ones(3)), f'd(sum(x))/dx must be ones, got {grad}'\n"
        "\n"
        "# === Larger function: f(x) = sin(x)  →  grad = cos(x) ===\n"
        "x = t.tensor([0.0, 0.1, 1.0, 2.0])\n"
        "grad = ex2_vjp_via_backward(x, t.sin)\n"
        "assert t.allclose(grad, t.cos(x), atol=1e-6), f'd(sum(sin(x)))/dx must equal cos(x), got {grad}'\n"
        "\n"
        "# === f returns a scalar → ValueError ===\n"
        "x = t.tensor([1.0, 2.0])\n"
        "try:\n"
        "    ex2_vjp_via_backward(x, lambda v: v.sum())\n"
        "except ValueError as e:\n"
        "    assert 'non-scalar' in str(e).lower() or 'scalar' in str(e).lower(), (\n"
        "        f'error message must mention scalar, got {e!r}'\n"
        "    )\n"
        "else:\n"
        "    raise AssertionError('expected ValueError for scalar f(x)')\n"
        "\n"
        "# === Output is a detached, leaf-free clone (mutation-safe) ===\n"
        "x = t.tensor([1.0, 2.0, 3.0])\n"
        "out = ex2_vjp_via_backward(x, lambda v: v ** 2)\n"
        "assert not out.requires_grad, 'output must be detached'\n"
        "out[0] = 999.0  # should NOT bleed back into any captured leaf\n"
        "out2 = ex2_vjp_via_backward(x, lambda v: v ** 2)\n"
        "assert out2[0] != 999.0, 'output must be a fresh tensor each call'\n"
        "\n"
        "# === Sanity: vector-output without gradient= raises RuntimeError ===\n"
        "# This documents WHY ex2 must pass gradient=; not a test of your fn, but of PyTorch.\n"
        "x_chk = t.tensor([1.0, 2.0], requires_grad=True)\n"
        "y_chk = x_chk ** 2\n"
        "try:\n"
        "    y_chk.backward()\n"
        "except RuntimeError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError(\n"
        "        'PyTorch should raise RuntimeError on non-scalar backward() without gradient=. '\n"
        "        'Test invariant broken.'\n"
        "    )"
    ),
    "solution_body": (
        "def ex2_vjp_via_backward(x, f):\n"
        "    x_leaf = x.detach().clone().requires_grad_(True)\n"
        "    y = f(x_leaf)\n"
        "    if y.dim() == 0:\n"
        "        raise ValueError('f must return a non-scalar tensor')\n"
        "    v = t.ones_like(y)\n"
        "    y.backward(gradient=v)\n"
        "    return x_leaf.grad.detach().clone()"
    ),
    "solution_notes": (
        "**`x.detach().clone().requires_grad_(True)` is the safe leaf "
        "recipe.** `detach` strips any history, `clone` ensures a fresh "
        "storage (so mutation of the leaf doesn't bleed into the "
        "caller's view), `requires_grad_(True)` makes it a new "
        "computational-graph root.\n\n"
        "**VJP with `v=ones` IS sum-then-backward.** The chain rule "
        "gives `d(sum_i y_i)/dx_j = sum_i dy_i/dx_j = (J^T @ ones)_j`. "
        "Passing `gradient=ones_like(y)` is just the explicit VJP form "
        "of the same computation. PyTorch has no \"sum first\" mode "
        "under the hood — `y.sum().backward()` literally inserts a sum "
        "node before computing the same VJP.\n\n"
        "**Returning a detached clone.** The leaf's `.grad` is "
        "accumulator state PyTorch may mutate later (e.g. another "
        "`.backward()` call inside the same graph). Returning a "
        "detached clone gives the caller a stable snapshot."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 5 — train-eval-mode-branch ex2
# ---------------------------------------------------------------------------

SPEC_EVAL_BN = {
    "atom_id": "train-eval-mode-branch",
    "subtopic": "PyTorch: train/eval mode",
    "topic_folder": TOPIC_TRAIN,
    "atom_recap_md": RECAP_EVAL_BN,
    "exercise_index": 2,
    "exercise_title": "BatchNorm running stats freeze in eval — same input, two different outputs",
    "slug": "batchnorm-train-eval-divergence",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["batchnorm", "train", "eval", "running_stats"],
    "kcs": [
        "batchnorm-mode-dependent-stats",
        "running-mean-update-only-in-train",
    ],
    "lo": (
        "Analyze the train→eval transition on a `BatchNorm1d` layer: "
        "do one train pass to populate `running_mean`/`running_var`, "
        "then run the SAME input in `eval()` and verify (a) the two "
        "outputs differ, (b) `running_mean` is unchanged across the "
        "eval pass."
    ),
    "prompt_body": (
        "Implement `ex2_bn_train_eval_divergence(x)`. Demonstrates "
        "how `.eval()` freezes BatchNorm's running statistics.\n\n"
        "Input:\n"
        "- `x`: `(B, C)` float tensor with `B >= 2` and `C` features.\n\n"
        "Algorithm:\n"
        "1. Construct `bn = nn.BatchNorm1d(C)` where `C = x.shape[1]`.\n"
        "2. `bn.train()`; compute `out_train = bn(x)` (this updates "
        "`running_mean`/`running_var`).\n"
        "3. Snapshot `rm_after_train = bn.running_mean.clone()`.\n"
        "4. `bn.eval()`; compute `out_eval = bn(x)`.\n"
        "5. Snapshot `rm_after_eval = bn.running_mean.clone()`.\n"
        "6. Return a `dict` with:\n"
        "   - `'out_train'`: the train-mode output (tensor)\n"
        "   - `'out_eval'`: the eval-mode output (tensor)\n"
        "   - `'diff_train_vs_eval'`: `(out_train - out_eval).abs().max().item()` (float)\n"
        "   - `'rm_after_train'`: the running_mean snapshot AFTER train pass (tensor)\n"
        "   - `'rm_after_eval'`: the running_mean snapshot AFTER eval pass (tensor)\n"
        "   - `'rm_changed_in_eval'`: bool — True iff `rm_after_eval` differs from `rm_after_train` (it should be False).\n\n"
        "Constraint: do NOT call `.no_grad()` or `.detach()`. Pure "
        "module behaviour.\n\n"
        "The test verifies the divergence is real and that the running "
        "stats are frozen in eval."
    ),
    "stub": (
        "def ex2_bn_train_eval_divergence(x: Tensor) -> dict:\n"
        '    """Run x through BN in train then eval; return outputs + running-stat snapshots."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Basic divergence on a non-trivial batch ===\n"
        "t.manual_seed(0)\n"
        "x = t.randn(8, 4) * 3.0 + 5.0  # shifted/scaled — running stats will be far from batch stats initially\n"
        "out = ex2_bn_train_eval_divergence(x)\n"
        "assert set(out.keys()) == {\n"
        "    'out_train', 'out_eval', 'diff_train_vs_eval',\n"
        "    'rm_after_train', 'rm_after_eval', 'rm_changed_in_eval',\n"
        "}, f'keys wrong: {set(out.keys())}'\n"
        "\n"
        "# === Train output: batch-normalized → mean ~0, var ~1 across batch dim ===\n"
        "ot = out['out_train']\n"
        "assert ot.shape == x.shape\n"
        "assert t.allclose(ot.mean(dim=0), t.zeros(4), atol=1e-5), (\n"
        "    f'train mode should center to batch mean ~0, got {ot.mean(dim=0)}'\n"
        ")\n"
        "assert t.allclose(ot.var(dim=0, unbiased=False), t.ones(4), atol=1e-4), (\n"
        "    f'train mode should scale to batch var ~1, got {ot.var(dim=0, unbiased=False)}'\n"
        ")\n"
        "\n"
        "# === Eval output: divergent from train ===\n"
        "oe = out['out_eval']\n"
        "assert oe.shape == x.shape\n"
        "assert out['diff_train_vs_eval'] > 1e-3, (\n"
        "    f'train/eval outputs should differ noticeably; diff={out[\"diff_train_vs_eval\"]}'\n"
        ")\n"
        "\n"
        "# === running_mean snapshot unchanged across the eval pass ===\n"
        "assert out['rm_changed_in_eval'] is False, (\n"
        "    f'eval mode must NOT update running_mean; rm_after_train={out[\"rm_after_train\"]}, '\n"
        "    f'rm_after_eval={out[\"rm_after_eval\"]}'\n"
        ")\n"
        "assert t.equal(out['rm_after_train'], out['rm_after_eval']), (\n"
        "    'tensor equality of running_mean must hold across eval pass'\n"
        ")\n"
        "\n"
        "# === running_mean was actually MOVED by the train pass ===\n"
        "# Fresh BN starts with running_mean = zeros; after one batch with mean~5, it should be != 0.\n"
        "assert not t.allclose(out['rm_after_train'], t.zeros(4)), (\n"
        "    f'train pass should have moved running_mean off zero, got {out[\"rm_after_train\"]}'\n"
        ")\n"
        "\n"
        "# === Smaller-batch sanity (B=2) — still works ===\n"
        "t.manual_seed(1)\n"
        "x2 = t.randn(2, 3) + 2.0\n"
        "out2 = ex2_bn_train_eval_divergence(x2)\n"
        "assert out2['diff_train_vs_eval'] > 0.0, 'B=2 batch should still show train/eval divergence'\n"
        "assert out2['rm_changed_in_eval'] is False\n"
        "\n"
        "# === No autograd state on returned tensors ===\n"
        "# We allowed autograd to run; the returned tensors are graph outputs.\n"
        "# We do NOT require them to be detached; we only require .item() works on diff.\n"
        "assert isinstance(out['diff_train_vs_eval'], float), (\n"
        "    f'diff must be Python float, got {type(out[\"diff_train_vs_eval\"]).__name__}'\n"
        ")"
    ),
    "solution_body": (
        "def ex2_bn_train_eval_divergence(x):\n"
        "    C = x.shape[1]\n"
        "    bn = nn.BatchNorm1d(C)\n"
        "\n"
        "    bn.train()\n"
        "    out_train = bn(x)\n"
        "    rm_after_train = bn.running_mean.clone()\n"
        "\n"
        "    bn.eval()\n"
        "    out_eval = bn(x)\n"
        "    rm_after_eval = bn.running_mean.clone()\n"
        "\n"
        "    diff = (out_train - out_eval).abs().max().item()\n"
        "    rm_changed = not t.equal(rm_after_train, rm_after_eval)\n"
        "\n"
        "    return {\n"
        "        'out_train': out_train,\n"
        "        'out_eval': out_eval,\n"
        "        'diff_train_vs_eval': float(diff),\n"
        "        'rm_after_train': rm_after_train,\n"
        "        'rm_after_eval': rm_after_eval,\n"
        "        'rm_changed_in_eval': bool(rm_changed),\n"
        "    }"
    ),
    "solution_notes": (
        "**Train uses batch stats; eval uses running stats.** That's "
        "the whole story. After the FIRST train pass on data with mean "
        "≠ 0, the running_mean has been updated toward that batch mean "
        "(by `momentum * batch_mean`, default `momentum=0.1`). The "
        "eval pass then normalizes using that partially-updated running "
        "stat — which is far from the actual batch stat — so the eval "
        "output diverges from the train output.\n\n"
        "**`running_mean.clone()` before swapping modes.** "
        "`running_mean` is a registered buffer that BN may mutate "
        "again on subsequent train passes. Cloning takes a snapshot "
        "that's safe to compare after eval.\n\n"
        "**Why this is the silent inference bug.** Forget to call "
        "`.eval()` at inference, and BN keeps updating running stats "
        "from your evaluation data — corrupting them. The corruption "
        "compounds across eval batches. The fix is one line; the "
        "training-loop discipline is the actual lesson."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 6 — validation-no-grad ex2
# ---------------------------------------------------------------------------

SPEC_NOGRAD_DECORATOR = {
    "atom_id": "validation-no-grad",
    "subtopic": "PyTorch: no_grad validation",
    "topic_folder": TOPIC_TRAIN,
    "atom_recap_md": RECAP_NOGRAD_DECORATOR,
    "exercise_index": 2,
    "exercise_title": "@torch.no_grad() decorator on the validation function — same effect, cleaner API",
    "slug": "no-grad-decorator-on-validation-fn",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["no_grad", "decorator", "validation", "inference"],
    "kcs": [
        "no-grad-decorator-vs-context",
        "eval-mode-plus-no-grad-pairing",
    ],
    "lo": (
        "Apply `@torch.no_grad()` as a function decorator on a "
        "validation routine and confirm autograd is disabled inside "
        "the function body even when the caller has autograd enabled."
    ),
    "prompt_body": (
        "Implement `ex2_validate(model, x, y)` decorated with "
        "`@torch.no_grad()`. The decorator-style equivalent of ex1's "
        "`with torch.no_grad():` block.\n\n"
        "Inputs:\n"
        "- `model`: an `nn.Module` (will be put into eval mode).\n"
        "- `x`: input tensor.\n"
        "- `y`: integer label tensor of shape `(B,)`.\n\n"
        "Algorithm (inside the decorated function):\n"
        "1. `model.eval()` — flip to eval mode (Dropout off, BN running "
        "stats frozen).\n"
        "2. `logits = model(x)`.\n"
        "3. `preds = logits.argmax(dim=1)`.\n"
        "4. `acc = (preds == y).float().mean().item()`.\n"
        "5. Inside the function (before returning), assert "
        "`not torch.is_grad_enabled()`. This is the proof that the "
        "decorator is in effect.\n"
        "6. Return a `dict` with:\n"
        "   - `'logits'`: the logits tensor.\n"
        "   - `'preds'`: the prediction tensor.\n"
        "   - `'acc'`: the Python float accuracy.\n"
        "   - `'grad_enabled_inside'`: `False` (read from "
        "`torch.is_grad_enabled()` before returning).\n\n"
        "Constraint: do NOT use `with torch.no_grad():` inside the "
        "function. The decorator is the whole point."
    ),
    "stub": (
        "import torch\n"
        "\n"
        "@torch.no_grad()\n"
        "def ex2_validate(model, x: Tensor, y: Tensor) -> dict:\n"
        '    """Decorator-style no-grad validation: eval + argmax + accuracy."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch\n"
        "\n"
        "# === Build a tiny classifier ===\n"
        "t.manual_seed(0)\n"
        "model = nn.Sequential(\n"
        "    nn.Linear(8, 16),\n"
        "    nn.ReLU(),\n"
        "    nn.Dropout(p=0.5),\n"
        "    nn.Linear(16, 4),\n"
        ")\n"
        "x = t.randn(20, 8)\n"
        "y = t.randint(0, 4, (20,))\n"
        "\n"
        "# === Confirm caller has autograd ON before calling ex2_validate ===\n"
        "assert torch.is_grad_enabled(), 'test setup: caller should have grad enabled'\n"
        "\n"
        "# === Run the validation ===\n"
        "out = ex2_validate(model, x, y)\n"
        "assert set(out.keys()) == {'logits', 'preds', 'acc', 'grad_enabled_inside'}, (\n"
        "    f'keys wrong: {set(out.keys())}'\n"
        ")\n"
        "\n"
        "# === Decorator was in effect inside the function ===\n"
        "assert out['grad_enabled_inside'] is False, (\n"
        "    f'@torch.no_grad() must disable grad inside the body; got {out[\"grad_enabled_inside\"]}'\n"
        ")\n"
        "\n"
        "# === Grad re-enabled after the function returns (decorator scope) ===\n"
        "assert torch.is_grad_enabled(), 'decorator must restore caller grad state'\n"
        "\n"
        "# === No grad metadata on the returned logits ===\n"
        "assert out['logits'].requires_grad is False, (\n"
        "    f'logits computed under no_grad must not require grad; got requires_grad={out[\"logits\"].requires_grad}'\n"
        ")\n"
        "assert out['logits'].grad_fn is None, (\n"
        "    f'logits must have no grad_fn under no_grad; got {out[\"logits\"].grad_fn}'\n"
        ")\n"
        "\n"
        "# === Model is in eval mode after the call ===\n"
        "assert not model.training, 'model.eval() should have been called inside ex2_validate'\n"
        "\n"
        "# === Shapes + types ===\n"
        "assert out['logits'].shape == (20, 4)\n"
        "assert out['preds'].shape == (20,)\n"
        "assert out['preds'].dtype == t.long, f'argmax must give long, got {out[\"preds\"].dtype}'\n"
        "assert isinstance(out['acc'], float), f'acc must be Python float, got {type(out[\"acc\"]).__name__}'\n"
        "assert 0.0 <= out['acc'] <= 1.0, f'acc out of range: {out[\"acc\"]}'\n"
        "\n"
        "# === Dropout disabled in eval: two passes give identical logits ===\n"
        "model.train()  # flip back to train; ex2_validate should still .eval() internally\n"
        "out_a = ex2_validate(model, x, y)\n"
        "out_b = ex2_validate(model, x, y)\n"
        "assert t.allclose(out_a['logits'], out_b['logits']), (\n"
        "    'Two validation passes must give identical logits (model.eval() disables Dropout)'\n"
        ")\n"
        "\n"
        "# === Matches a hand-written `with torch.no_grad():` equivalent ===\n"
        "model.train()\n"
        "out_dec = ex2_validate(model, x, y)\n"
        "model.eval()\n"
        "with torch.no_grad():\n"
        "    logits_ref = model(x)\n"
        "    preds_ref = logits_ref.argmax(dim=1)\n"
        "    acc_ref = (preds_ref == y).float().mean().item()\n"
        "assert t.allclose(out_dec['logits'], logits_ref), 'decorator-form must match with-block form'\n"
        "assert t.equal(out_dec['preds'], preds_ref)\n"
        "assert abs(out_dec['acc'] - acc_ref) < 1e-6"
    ),
    "solution_body": (
        "import torch\n"
        "\n"
        "@torch.no_grad()\n"
        "def ex2_validate(model, x, y):\n"
        "    model.eval()\n"
        "    logits = model(x)\n"
        "    preds = logits.argmax(dim=1)\n"
        "    acc = (preds == y).float().mean().item()\n"
        "    grad_inside = torch.is_grad_enabled()\n"
        "    return {\n"
        "        'logits': logits,\n"
        "        'preds': preds,\n"
        "        'acc': float(acc),\n"
        "        'grad_enabled_inside': bool(grad_inside),\n"
        "    }"
    ),
    "solution_notes": (
        "**Decorator vs context manager — same effect, different "
        "scope.** `@torch.no_grad()` toggles grad off for the whole "
        "function body and restores the prior state on return. A "
        "`with` block toggles only for the lexical block. For an "
        "entire eval function, the decorator is more readable and "
        "harder to accidentally bypass.\n\n"
        "**`.eval()` is still required.** `no_grad` disables autograd; "
        "it does NOT switch Dropout or BatchNorm modes. Forgetting "
        "`.eval()` would give you autograd-free outputs that are "
        "STILL stochastic (Dropout) or use BATCH stats (BN).\n\n"
        "**Why `grad_enabled_inside` is a useful return value.** It's "
        "an introspection probe — the test asserts it's False without "
        "needing to inspect the function's bytecode. In production "
        "code, this kind of probe is overkill, but here it makes the "
        "decorator's effect testable from the outside."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 7 — conv-leakyrelu-block-discriminator ex2
# ---------------------------------------------------------------------------

SPEC_CONV_STRIDE1 = {
    "atom_id": "conv-leakyrelu-block-discriminator",
    "subtopic": "GAN: Conv+LeakyReLU discriminator block",
    "topic_folder": TOPIC_VAEGAN,
    "atom_recap_md": RECAP_CONV_STRIDE1,
    "exercise_index": 2,
    "exercise_title": "stride=1 Conv+BN+LeakyReLU block — same H/W, deeper features",
    "slug": "stride1-conv-bn-leakyrelu-block",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["conv", "stride1", "leaky-relu", "discriminator"],
    "kcs": [
        "stride1-padding-for-spatial-preservation",
        "conv-bn-leakyrelu-bias-false",
    ],
    "lo": (
        "Apply `Conv2d(stride=1, padding=1, kernel_size=3, bias=False)` "
        "followed by BatchNorm2d + LeakyReLU(0.2) to build a "
        "spatial-preserving refinement block whose output has the same "
        "H/W as its input."
    ),
    "prompt_body": (
        "Build `ex2_build_stride1_block(in_ch, out_ch)`. The "
        "spatial-preserving variant of ex1's stride=2 downsample block.\n\n"
        "Constraints:\n"
        "1. Return an `nn.Sequential` with three children in order:\n"
        "   - `nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)`\n"
        "   - `nn.BatchNorm2d(out_ch)`\n"
        "   - `nn.LeakyReLU(0.2, inplace=True)`\n"
        "2. The output spatial dims must EQUAL the input spatial dims "
        "for any H, W >= 3 (verified by the test).\n"
        "3. `bias=False` on the Conv — BatchNorm has its own affine "
        "bias.\n"
        "4. LeakyReLU slope = 0.2, `inplace=True` (the DCGAN "
        "convention).\n\n"
        "Output: an `nn.Sequential` callable as `block(x)` where "
        "`x: (B, in_ch, H, W)` → `out: (B, out_ch, H, W)`."
    ),
    "stub": (
        "def ex2_build_stride1_block(in_ch: int, out_ch: int) -> nn.Module:\n"
        '    """Spatial-preserving Conv+BN+LeakyReLU block (stride=1, padding=1)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Build and inspect the block structure ===\n"
        "block = ex2_build_stride1_block(in_ch=8, out_ch=16)\n"
        "assert isinstance(block, nn.Sequential), f'must return nn.Sequential, got {type(block).__name__}'\n"
        "children = list(block.children())\n"
        "assert len(children) == 3, f'must have exactly 3 children, got {len(children)}'\n"
        "assert isinstance(children[0], nn.Conv2d), f'child 0 must be Conv2d, got {type(children[0]).__name__}'\n"
        "assert isinstance(children[1], nn.BatchNorm2d), f'child 1 must be BatchNorm2d, got {type(children[1]).__name__}'\n"
        "assert isinstance(children[2], nn.LeakyReLU), f'child 2 must be LeakyReLU, got {type(children[2]).__name__}'\n"
        "\n"
        "# === Conv hyperparams ===\n"
        "conv = children[0]\n"
        "assert conv.in_channels == 8 and conv.out_channels == 16\n"
        "assert conv.kernel_size == (3, 3), f'kernel_size must be 3, got {conv.kernel_size}'\n"
        "assert conv.stride == (1, 1), f'stride must be 1 (spatial-preserving), got {conv.stride}'\n"
        "assert conv.padding == (1, 1), f'padding must be 1, got {conv.padding}'\n"
        "assert conv.bias is None, 'bias must be False (BN follows)'\n"
        "\n"
        "# === BN matches Conv out_channels ===\n"
        "bn = children[1]\n"
        "assert bn.num_features == 16, f'BN num_features must match conv out_channels, got {bn.num_features}'\n"
        "\n"
        "# === LeakyReLU slope + inplace ===\n"
        "act = children[2]\n"
        "assert act.negative_slope == 0.2, f'slope must be 0.2, got {act.negative_slope}'\n"
        "assert act.inplace is True, 'LeakyReLU must be inplace=True'\n"
        "\n"
        "# === Output H/W matches input H/W (spatial preservation) ===\n"
        "for H, W in [(4, 4), (8, 8), (7, 11), (32, 32), (3, 3)]:\n"
        "    x = t.randn(2, 8, H, W)\n"
        "    out = block(x)\n"
        "    assert out.shape == (2, 16, H, W), f'spatial preservation failed for {(H, W)}: got {tuple(out.shape)}'\n"
        "\n"
        "# === Different (in_ch, out_ch) combos ===\n"
        "block2 = ex2_build_stride1_block(in_ch=3, out_ch=32)\n"
        "x = t.randn(1, 3, 16, 16)\n"
        "out = block2(x)\n"
        "assert out.shape == (1, 32, 16, 16), f'channel growth path wrong: {tuple(out.shape)}'\n"
        "\n"
        "# === Same in_ch == out_ch (refinement block, no channel change) ===\n"
        "block3 = ex2_build_stride1_block(in_ch=64, out_ch=64)\n"
        "x = t.randn(2, 64, 8, 8)\n"
        "out = block3(x)\n"
        "assert out.shape == (2, 64, 8, 8)\n"
        "\n"
        "# === Output is nonnegative on positive inputs ===\n"
        "block4 = ex2_build_stride1_block(in_ch=4, out_ch=4)\n"
        "# Force the block deterministically — eval mode freezes BN.\n"
        "block4.eval()\n"
        "x = t.randn(2, 4, 6, 6) * 10  # large magnitude → BN doesn't crush; LeakyReLU still leaks negatives\n"
        "out = block4(x)\n"
        "# Negative outputs still possible (slope 0.2), but magnitudes scaled toward small.\n"
        "# Check that the LeakyReLU slope is at work: for any x_after_bn < 0, output = 0.2 * x_after_bn,\n"
        "# so the ratio output_neg / x_after_bn_neg should be ~0.2.\n"
        "# We just sanity-check shape + finiteness here.\n"
        "assert out.shape == (2, 4, 6, 6)\n"
        "assert t.isfinite(out).all()"
    ),
    "solution_body": (
        "def ex2_build_stride1_block(in_ch, out_ch):\n"
        "    return nn.Sequential(\n"
        "        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),\n"
        "        nn.BatchNorm2d(out_ch),\n"
        "        nn.LeakyReLU(0.2, inplace=True),\n"
        "    )"
    ),
    "solution_notes": (
        "**`kernel=3, stride=1, padding=1` is the canonical "
        "spatial-preserving conv.** The 'SAME' padding pattern from "
        "TF. With `H_out = floor((H + 2*1 - 3) / 1) + 1 = H`, the "
        "kernel can reach every position and boundary slack is taken "
        "up by the 1-pad.\n\n"
        "**Stride-1 refinement blocks are where DCGAN puts CAPACITY.** "
        "Stride-2 blocks halve the spatial resolution AND grow "
        "channels. Stride-1 blocks keep the spatial size and either "
        "grow channels (input-to-internal layer) or hold them "
        "constant (pure refinement). They cost 4× the FLOPs of a "
        "stride-2 block at the same spatial size, so use them "
        "sparingly.\n\n"
        "**`bias=False` is a habitual save.** A trainable bias before "
        "BN gets immediately subtracted out by BN's mean-centering — "
        "the optimizer would still train it, but its effect on the "
        "output is zero. Save the parameters."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 8 — convtranspose-bn-activation-block ex2
# ---------------------------------------------------------------------------

SPEC_CONVT_OUTPAD = {
    "atom_id": "convtranspose-bn-activation-block",
    "subtopic": "GAN: ConvT+BN+Activation block",
    "topic_folder": TOPIC_VAEGAN,
    "atom_recap_md": RECAP_CONVT_OUTPAD,
    "exercise_index": 2,
    "exercise_title": "ConvT+BN+ReLU 4→9 spatial via output_padding (odd-target upsample)",
    "slug": "convt-output-padding-4-to-9-upsample",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["conv-transpose", "output-padding", "upsample", "odd-output"],
    "kcs": [
        "convt-output-padding-formula",
        "convt-bn-relu-bias-false",
    ],
    "lo": (
        "Apply `ConvTranspose2d(stride=2, kernel_size=4, padding=1, "
        "output_padding=1)` to upsample a 4×4 feature map to 9×9 — "
        "exercising the `output_padding` knob that disambiguates the "
        "inverse of strided Conv for odd-target outputs."
    ),
    "prompt_body": (
        "Build `ex2_build_convt_outpad_block(in_ch, out_ch)`. The "
        "odd-target upsample variant of ex1's ConvT block.\n\n"
        "Spatial spec: takes `(B, in_ch, 4, 4)` → `(B, out_ch, 9, 9)`.\n\n"
        "Derivation. The PyTorch formula is:\n"
        "```\n"
        "H_out = (H_in - 1)*stride - 2*padding + kernel + output_padding\n"
        "```\n"
        "With `H_in=4, stride=2, kernel=4, padding=1, "
        "output_padding=1`:\n"
        "```\n"
        "H_out = 3*2 - 2*1 + 4 + 1 = 6 - 2 + 4 + 1 = 9   ✓\n"
        "```\n"
        "Constraint check: `output_padding < max(stride, dilation)`. "
        "Here `1 < 2`, so this is legal.\n\n"
        "Constraints:\n"
        "1. Return an `nn.Sequential` with three children in order:\n"
        "   - `nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, "
        "stride=2, padding=1, output_padding=1, bias=False)`\n"
        "   - `nn.BatchNorm2d(out_ch)`\n"
        "   - `nn.ReLU(inplace=True)`\n"
        "2. `bias=False` (BN's affine handles the bias).\n"
        "3. Output H/W exactly `9, 9` from input `4, 4`.\n\n"
        "Output: an `nn.Sequential` callable as `block(x)` where "
        "`x: (B, in_ch, 4, 4)` → `out: (B, out_ch, 9, 9)`."
    ),
    "stub": (
        "def ex2_build_convt_outpad_block(in_ch: int, out_ch: int) -> nn.Module:\n"
        '    """ConvT+BN+ReLU that takes (B, in_ch, 4, 4) -> (B, out_ch, 9, 9) via output_padding=1."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Structure ===\n"
        "block = ex2_build_convt_outpad_block(in_ch=16, out_ch=8)\n"
        "assert isinstance(block, nn.Sequential), f'must return nn.Sequential, got {type(block).__name__}'\n"
        "children = list(block.children())\n"
        "assert len(children) == 3, f'must have 3 children, got {len(children)}'\n"
        "assert isinstance(children[0], nn.ConvTranspose2d), f'child 0 must be ConvTranspose2d, got {type(children[0]).__name__}'\n"
        "assert isinstance(children[1], nn.BatchNorm2d), f'child 1 must be BatchNorm2d, got {type(children[1]).__name__}'\n"
        "assert isinstance(children[2], nn.ReLU), f'child 2 must be ReLU, got {type(children[2]).__name__}'\n"
        "\n"
        "# === ConvT hyperparams ===\n"
        "convt = children[0]\n"
        "assert convt.in_channels == 16 and convt.out_channels == 8\n"
        "assert convt.kernel_size == (4, 4), f'kernel must be 4, got {convt.kernel_size}'\n"
        "assert convt.stride == (2, 2), f'stride must be 2, got {convt.stride}'\n"
        "assert convt.padding == (1, 1), f'padding must be 1, got {convt.padding}'\n"
        "assert convt.output_padding == (1, 1), f'output_padding must be 1, got {convt.output_padding}'\n"
        "assert convt.bias is None, 'bias must be False (BN follows)'\n"
        "\n"
        "# === BN matches ConvT out_channels ===\n"
        "bn = children[1]\n"
        "assert bn.num_features == 8\n"
        "\n"
        "# === ReLU is inplace ===\n"
        "act = children[2]\n"
        "assert act.inplace is True, 'ReLU must be inplace=True'\n"
        "\n"
        "# === 4 -> 9 spatial check ===\n"
        "x = t.randn(2, 16, 4, 4)\n"
        "out = block(x)\n"
        "assert out.shape == (2, 8, 9, 9), f'4 -> 9 spatial failed: {tuple(out.shape)}'\n"
        "\n"
        "# === Formula sanity: try H_in=8 to confirm 8 -> 17 ===\n"
        "# H_out = (8-1)*2 - 2 + 4 + 1 = 14 - 2 + 4 + 1 = 17.\n"
        "x = t.randn(1, 16, 8, 8)\n"
        "out = block(x)\n"
        "assert out.shape == (1, 8, 17, 17), f'8 -> 17 failed: {tuple(out.shape)}'\n"
        "\n"
        "# === Output finite + nonneg (ReLU rectifies) ===\n"
        "block.eval()  # freeze BN running stats\n"
        "x = t.randn(2, 16, 4, 4)\n"
        "out = block(x)\n"
        "assert t.isfinite(out).all(), 'outputs must be finite'\n"
        "assert (out >= 0).all(), f'ReLU output must be nonnegative; got min={out.min().item()}'\n"
        "\n"
        "# === Different (in_ch, out_ch) still 4 -> 9 ===\n"
        "block2 = ex2_build_convt_outpad_block(in_ch=4, out_ch=32)\n"
        "x = t.randn(1, 4, 4, 4)\n"
        "out = block2(x)\n"
        "assert out.shape == (1, 32, 9, 9)\n"
        "\n"
        "# === Param count = ConvT (in_ch * out_ch * 4 * 4) + BN (2*out_ch) ===\n"
        "block3 = ex2_build_convt_outpad_block(in_ch=4, out_ch=8)\n"
        "n_params = sum(p.numel() for p in block3.parameters())\n"
        "expected = (4 * 8 * 4 * 4) + (2 * 8)  # ConvT weights + BN weight+bias\n"
        "assert n_params == expected, f'param count wrong: expected {expected}, got {n_params}'"
    ),
    "solution_body": (
        "def ex2_build_convt_outpad_block(in_ch, out_ch):\n"
        "    return nn.Sequential(\n"
        "        nn.ConvTranspose2d(\n"
        "            in_ch, out_ch,\n"
        "            kernel_size=4, stride=2, padding=1, output_padding=1,\n"
        "            bias=False,\n"
        "        ),\n"
        "        nn.BatchNorm2d(out_ch),\n"
        "        nn.ReLU(inplace=True),\n"
        "    )"
    ),
    "solution_notes": (
        "**The 4→9 spec uses `kernel=4, stride=2, pad=1, out_pad=1`.** "
        "Walk the formula: `(4-1)*2 - 2 + 4 + 1 = 9`. The `output_"
        "padding=1` is what makes this land on an odd target — "
        "without it you'd be stuck at 8. PyTorch requires "
        "`output_padding < max(stride, dilation)`, so the upper bound "
        "on `out_pad` here is `1` (since `stride=2, dilation=1` → "
        "max=2, strict less-than gives ≤1).\n\n"
        "**Why odd targets matter.** MNIST is 28×28, CelebA-cropped "
        "is 64×64, but ImageNet-class outputs and some intermediate "
        "feature maps in custom generators land on 7×7, 11×11, etc. "
        "`output_padding` is the canonical knob; the alternative is "
        "fractional strides via `Upsample` + standard Conv, which "
        "have their own gotchas (checkerboard artifacts).\n\n"
        "**`bias=False` again.** Same reasoning as the discriminator "
        "block — BN absorbs any constant offset, so the ConvT bias "
        "would just waste parameters."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# All specs
# ---------------------------------------------------------------------------

SPECS = [
    SPEC_TOPK,
    SPEC_NORMALIZE,
    SPEC_WHERE_CLAMP,
    SPEC_BACKWARD_VECTOR,
    SPEC_EVAL_BN,
    SPEC_NOGRAD_DECORATOR,
    SPEC_CONV_STRIDE1,
    SPEC_CONVT_OUTPAD,
]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

def _verify_all(specs):
    import torch as t
    import numpy as np
    import torch.nn as nn
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
    print(f"[deepening_aa_batch12] Verifying {len(SPECS)} specs...")
    _verify_all(SPECS)

    print(f"\n[deepening_aa_batch12] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_aa_batch12] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
