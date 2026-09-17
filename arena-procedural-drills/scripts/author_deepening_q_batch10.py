#!/usr/bin/env python3
"""Author 8 ex2 deepening drills (batch 10).

Atoms (5 DCGAN + 3 distributed):
    - dcgan-normal-init-002              (ex2: gain-scaled normal init + named-modules report)
    - generator-loss-fool-discriminator  (ex2: logits-form non-saturating G loss via BCE-with-logits)
    - model-train-eval-toggle-around-sample (ex2: contextmanager helper for exception-safe eval toggle)
    - module-modules-iter-isinstance-dispatch (ex2: named_modules + dotted-name dispatch report)
    - noise-batch-from-latent            (ex2: unit-sphere-normalized noise on a target device)
    - all-reduce-compose                 (ex2: compose all_reduce with MAX op via reduce + broadcast)
    - all-reduce-eval-metrics            (ex2: sample-count-weighted eval mean via two all_reduces)
    - all-reduce-grad-sync               (ex2: skip-sync optimization — only sync params with grad != None)

Each ex2 hits a DISTINCT facet from ex1: different cognitive operation, surface
context, or numerical regime. ONE LO + ONE Bloom + <=2 KCs per drill.

Distributed atoms cannot really run `torch.distributed` in the CPU-only
verifier — we mock `dist.all_reduce` / `dist.reduce` / `dist.broadcast` via
a thread-driven `_FakeDist` simulator (N threads share a barrier, all_reduce
collects per-rank tensors and re-emits the reduced value). Mocks are
installed inside each test_body via unittest.mock.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC_DCGAN = "prereqs_dcgan_final"
TOPIC_DIST = "prereqs_distributed"


# ---------------------------------------------------------------------------
# Recap blocks (deepening-focused — kept tight, build on ex1's recap).
# ---------------------------------------------------------------------------

RECAP_NORMAL_INIT_DEEP = (
    "## DCGAN normal init — named-module variant\n"
    "\n"
    "Ex1 walked layers via `model.apply` (a pure transform).  Here we want a "
    "**report**: which named submodule got which init, by dotted path. The "
    "tool is `model.named_modules()` — same recursion as `modules()`, but "
    "each yielded item is `(qualified_name: str, module)`:\n"
    "\n"
    "```python\n"
    "for name, m in model.named_modules():\n"
    "    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):\n"
    "        nn.init.normal_(m.weight, mean=0.0, std=0.02)\n"
    "        records.append((name, 'conv', m.weight.std().item()))\n"
    "```\n"
    "\n"
    "**Why named_modules over modules.** When you need to LOG what you "
    "initialized (which is most real-world training-script init code), you "
    "want the qualified name (`'features.3.conv'`) — not just the type. "
    "`apply` gives you neither.\n"
    "\n"
    "**Why also gain-scale.** Production DCGAN code sometimes scales the "
    "std by a `gain` factor per layer type (e.g. ConvTranspose gets `gain` "
    "× 0.02 to compensate for the upsample). The ex2 drills the parametric "
    "form."
)

RECAP_G_LOSS_LOGITS = (
    "## Generator loss — logits form (numerically stable)\n"
    "\n"
    "Ex1 used `F.binary_cross_entropy(d_pred_prob, ones_like)` — the bare-"
    "probability form. In production, D's last layer is typically a Linear / "
    "Conv whose OUTPUT is a logit; sigmoid is fused into the loss:\n"
    "\n"
    "```python\n"
    "d_logits = D(fakes)                              # raw, no sigmoid\n"
    "loss_G   = F.binary_cross_entropy_with_logits(\n"
    "    d_logits, t.ones_like(d_logits)\n"
    ")\n"
    "```\n"
    "\n"
    "**Why `_with_logits`.** Numerically stable. `log(sigmoid(z))` and "
    "`log(1 - sigmoid(z))` both lose precision when `|z|` is large; the "
    "fused form uses `log1p(exp(-z))` and stays finite up to `|z| ≈ 80`.\n"
    "\n"
    "**Same target=1 trick as ex1.** The non-saturating G loss still uses "
    "target 1 on fakes — only the loss FUNCTION changes (logits vs probs)."
)

RECAP_TRAIN_EVAL_CTX = (
    "## `eval()` as a context manager — exception-safe restore\n"
    "\n"
    "Ex1 used the bare `eval() → no_grad → forward → train()` block. The "
    "fragile bit: if the forward raises (OOM, NaN, etc.) `train()` never "
    "runs and the model is stuck in eval mode for the rest of the loop. The "
    "context-manager wrap fixes that with `try/finally`:\n"
    "\n"
    "```python\n"
    "from contextlib import contextmanager\n"
    "\n"
    "@contextmanager\n"
    "def eval_mode(model):\n"
    "    was_training = model.training\n"
    "    model.eval()\n"
    "    try:\n"
    "        with t.no_grad():\n"
    "            yield model\n"
    "    finally:\n"
    "        if was_training:\n"
    "            model.train()\n"
    "```\n"
    "\n"
    "**Why capture `was_training`.** If the caller is already in eval mode "
    "(e.g. inside a nested `eval_mode`), you must NOT flip to train on exit "
    "— restore the prior state. The pre-check is one line and prevents the "
    "nested-call footgun.\n"
    "\n"
    "**Why `finally`.** A raised exception during forward (`NaN`, "
    "`CUDA OOM`, asserts) would skip the restore in a try/except. `finally` "
    "guarantees the toggle, even when re-raising."
)

RECAP_MODULES_NAMED = (
    "## `named_modules()` + dotted-name dispatch\n"
    "\n"
    "Ex1 counted layers by type — types only, no names. Real training-script "
    "instrumentation usually wants the QUALIFIED NAME plus the type:\n"
    "\n"
    "```python\n"
    "for qname, m in model.named_modules():\n"
    "    if isinstance(m, nn.Conv2d):\n"
    "        report[qname] = 'conv2d'\n"
    "```\n"
    "\n"
    "The qname looks like `'features.0'` or `'encoder.block1.bn'`. Dots are "
    "module-attribute names; numbers are Sequential indices.\n"
    "\n"
    "**Empty string for the root module.** `named_modules()` yields "
    "`('', model)` first — the root has no qualified name. Most reporting "
    "code filters that out (it's the container, not a meaningful submodule).\n"
    "\n"
    "**Why dotted names over plain types.** Two BatchNorm2d layers in a "
    "ResNet block need to be distinguishable in a layer-wise LR schedule, "
    "freezing schedule, or weight-decay-exclusion list. The qname is the "
    "unique identifier."
)

RECAP_NOISE_SPHERE = (
    "## Noise batch on a target device + unit-sphere normalization\n"
    "\n"
    "Ex1 built `(B, L, 1, 1)` standard-normal noise. Two production "
    "extensions:\n"
    "\n"
    "1. **Build directly on a target device** — `device=` kwarg saves a "
    "later `.to(device)` (one host→GPU copy, often the slowest step).\n"
    "2. **Unit-sphere normalization** — divide each per-sample vector by "
    "its L2 norm so latent codes lie on the unit hypersphere. Used in "
    "StyleGAN, BigGAN, and any setup where you want bounded latent "
    "interpolation:\n"
    "\n"
    "```python\n"
    "noise = t.randn(B, L, 1, 1, device=device, generator=g)\n"
    "norms = noise.flatten(1).norm(dim=1)         # (B,)\n"
    "noise = noise / norms.view(B, 1, 1, 1).clamp_min(1e-8)\n"
    "```\n"
    "\n"
    "**Why clamp the norm.** Theoretically the norm of `N(0, I_L)` is never "
    "zero, but at low `L` you can hit very small norms numerically. The "
    "`clamp_min(1e-8)` makes the divide safe even with `L=2`.\n"
    "\n"
    "**Sphere prior vs Gaussian prior.** Both train; the sphere prior has "
    "smoother interpolations (no probability mass near origin) but tighter "
    "support — your G must adapt."
)

RECAP_ALL_REDUCE_MAX = (
    "## `all_reduce` composition with MAX (not SUM)\n"
    "\n"
    "Ex1 composed `all_reduce(SUM) = reduce(SUM) + broadcast`. The "
    "composition is OP-PARAMETRIC — the exact same shape works for MAX, "
    "MIN, PRODUCT:\n"
    "\n"
    "```python\n"
    "def all_reduce_max(tensor, rank, world_size):\n"
    "    dist.reduce(tensor, dst=0, op=dist.ReduceOp.MAX)   # rank-0 gets the max\n"
    "    dist.broadcast(tensor, src=0)                       # everyone learns it\n"
    "```\n"
    "\n"
    "**Where MAX-all-reduce shows up.** Global max-norm gradient clipping, "
    "early-stop criteria ('any rank diverged?'), max sequence length per "
    "batch in dynamic batching. SUM is the most common, but MAX is "
    "non-negligible in real DDP code.\n"
    "\n"
    "**`dist.ReduceOp` is just an enum.** `SUM`, `PRODUCT`, `MAX`, `MIN`, "
    "`BAND`, `BOR`, `BXOR`, `PREMUL_SUM`. The reducer dispatches internally — "
    "your wrapper only changes the `op=` kwarg."
)

RECAP_ALL_REDUCE_WEIGHTED = (
    "## Sample-count-weighted eval mean (two all_reduces)\n"
    "\n"
    "Ex1 averaged a SCALAR loss with `all_reduce(SUM) / world_size`. That "
    "assumes every rank evaluated the SAME number of samples. In real "
    "distributed eval, the last batch is short — rank 3 might see 13 "
    "samples while ranks 0–2 see 32 each. Naive `mean` over-weights rank 3.\n"
    "\n"
    "Correct form: reduce `(sum_loss, count)` separately, then divide:\n"
    "\n"
    "```python\n"
    "stats = t.tensor([local_loss_sum, local_count], dtype=t.float32)\n"
    "dist.all_reduce(stats, op=dist.ReduceOp.SUM)\n"
    "global_mean = stats[0] / stats[1]\n"
    "```\n"
    "\n"
    "**Why one tensor, not two all_reduces.** Bandwidth — one network "
    "round-trip vs two. The two scalars get packed into a length-2 tensor "
    "and reduced together. Identical math result, half the latency.\n"
    "\n"
    "**Trap.** `local_loss_sum` (NOT `local_mean`). If you reduce the "
    "per-rank MEAN you lose the count weight and we're back to ex1's bug. "
    "The numerator must be the unreduced sum."
)

RECAP_GRAD_SYNC_SKIP = (
    "## Skip-sync optimization — only all_reduce non-None grads\n"
    "\n"
    "Ex1 looped every parameter and all_reduced its `.grad`. In real "
    "training, some parameters get NO gradient on this step:\n"
    "\n"
    "- Frozen layers (`requires_grad=False`).\n"
    "- Sparse models where this batch didn't touch certain heads.\n"
    "- Embedding tables when none of this batch's tokens hit them.\n"
    "\n"
    "Their `.grad` is `None`. Calling `dist.all_reduce(None, ...)` either "
    "crashes or silently sends a zero tensor — wasted bandwidth. The "
    "skip-sync optimization:\n"
    "\n"
    "```python\n"
    "for p in model.parameters():\n"
    "    if p.grad is None:\n"
    "        continue\n"
    "    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)\n"
    "    p.grad /= world_size\n"
    "```\n"
    "\n"
    "**Subtlety: rank consistency.** Every rank must agree on WHICH "
    "parameters to skip — if rank 0 skips `embedding.weight` but rank 1 "
    "does not, you deadlock (rank 1 hangs waiting for rank 0's all_reduce "
    "that never comes). In practice every rank runs the same code over the "
    "same model graph, so the `is None` check returns the same answer "
    "everywhere. If your grads diverge across ranks before sync, that's a "
    "bug ABOVE this layer.\n"
    "\n"
    "**Why not iterate `model.named_parameters()` instead.** Same answer; "
    "`.parameters()` is the canonical form and you don't need the names "
    "for this op."
)


# ---------------------------------------------------------------------------
# Fake-distributed harness — installed inside distributed test_bodies.
#
# Strategy: spawn `world_size` daemon THREADS sharing a barrier + a per-op
# scratchpad. The mocked `dist.all_reduce`, `dist.reduce`, `dist.broadcast`,
# and `dist.init_process_group` / `destroy_process_group` operate on the
# scratchpad. Each thread's `dist.get_rank()` returns the thread's pinned
# rank (kept in a thread-local).
#
# This block gets prepended to every distributed test_body. It defines:
#     _run_fake_world(worker_fn, world_size, *extra_args) -> list[result]
# which returns the per-rank return value (or whatever the worker writes to
# its result slot via `_FAKE_RESULTS[rank] = ...`).
# ---------------------------------------------------------------------------

_FAKE_DIST_HARNESS = r'''
import threading
import contextlib
import types as _types
from unittest.mock import patch
import torch as _t_for_fake
import torch.distributed as _dist_real

class _FakeReduceOp:
    SUM = 'SUM'
    MAX = 'MAX'
    MIN = 'MIN'
    PRODUCT = 'PROD'

class _FakeWorld:
    """Shared state across `world_size` rank-threads."""
    def __init__(self, world_size):
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size)
        self.lock = threading.Lock()
        # scratch[op_id] -> list of (rank, tensor); reset per op via barrier
        self.scratch = {}
        # per-rank thread-local pinned rank
        self.tls = threading.local()
        # collected per-rank results (for the test to read)
        self.results = [None] * world_size
    def all_reduce(self, tensor, op='SUM'):
        rank = self.tls.rank
        # phase 1: every rank deposits its tensor copy
        self.barrier.wait()
        with self.lock:
            self.scratch.setdefault('ar', [None] * self.world_size)
            self.scratch['ar'][rank] = tensor.detach().clone()
        self.barrier.wait()
        # phase 2: every rank reads-out the reduced result (same math)
        bag = self.scratch['ar']
        if op == 'SUM':
            reduced = bag[0].clone()
            for x in bag[1:]:
                reduced = reduced + x
        elif op == 'MAX':
            reduced = bag[0].clone()
            for x in bag[1:]:
                reduced = _t_for_fake.maximum(reduced, x)
        elif op == 'MIN':
            reduced = bag[0].clone()
            for x in bag[1:]:
                reduced = _t_for_fake.minimum(reduced, x)
        elif op == 'PROD':
            reduced = bag[0].clone()
            for x in bag[1:]:
                reduced = reduced * x
        else:
            raise ValueError(f'unknown fake op {op!r}')
        # mutate in-place so caller's tensor reflects the reduction
        tensor.copy_(reduced)
        self.barrier.wait()
        if rank == 0:
            self.scratch.pop('ar', None)
        self.barrier.wait()
    def reduce(self, tensor, dst, op='SUM'):
        rank = self.tls.rank
        self.barrier.wait()
        with self.lock:
            self.scratch.setdefault('rd', [None] * self.world_size)
            self.scratch['rd'][rank] = tensor.detach().clone()
        self.barrier.wait()
        # only the dst rank gets the reduced result
        if rank == dst:
            bag = self.scratch['rd']
            if op == 'SUM':
                reduced = bag[0].clone()
                for x in bag[1:]:
                    reduced = reduced + x
            elif op == 'MAX':
                reduced = bag[0].clone()
                for x in bag[1:]:
                    reduced = _t_for_fake.maximum(reduced, x)
            elif op == 'MIN':
                reduced = bag[0].clone()
                for x in bag[1:]:
                    reduced = _t_for_fake.minimum(reduced, x)
            elif op == 'PROD':
                reduced = bag[0].clone()
                for x in bag[1:]:
                    reduced = reduced * x
            else:
                raise ValueError(f'unknown fake op {op!r}')
            tensor.copy_(reduced)
        self.barrier.wait()
        if rank == 0:
            self.scratch.pop('rd', None)
        self.barrier.wait()
    def broadcast(self, tensor, src):
        rank = self.tls.rank
        self.barrier.wait()
        if rank == src:
            with self.lock:
                self.scratch['bc'] = tensor.detach().clone()
        self.barrier.wait()
        if rank != src:
            tensor.copy_(self.scratch['bc'])
        self.barrier.wait()
        if rank == 0:
            self.scratch.pop('bc', None)
        self.barrier.wait()
    def barrier_op(self):
        self.barrier.wait()

def _run_fake_world(worker_fn, world_size, *extra_args, timeout=30):
    world = _FakeWorld(world_size)
    errors = [None] * world_size
    def _runner(rank):
        world.tls.rank = rank
        # Build the fake `dist` module facade.
        fake_dist = _types.SimpleNamespace()
        fake_dist.ReduceOp = _FakeReduceOp
        fake_dist.all_reduce = lambda tensor, op='SUM': world.all_reduce(tensor, op)
        fake_dist.reduce = lambda tensor, dst, op='SUM': world.reduce(tensor, dst, op)
        fake_dist.broadcast = lambda tensor, src: world.broadcast(tensor, src)
        fake_dist.barrier = world.barrier_op
        fake_dist.get_rank = lambda: rank
        fake_dist.get_world_size = lambda: world_size
        fake_dist.init_process_group = lambda **kw: None
        fake_dist.destroy_process_group = lambda: None
        # Inject into the worker's calling globals.
        # The student code calls `dist.<op>`; we patch the `dist` name
        # in the calling namespace via direct globals injection.
        try:
            worker_fn(rank, world_size, fake_dist, world)
        except BaseException as e:
            import traceback as _tb
            errors[rank] = (e, _tb.format_exc())
    threads = [threading.Thread(target=_runner, args=(r,), daemon=True) for r in range(world_size)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=timeout)
    for r, err in enumerate(errors):
        if err is not None:
            raise RuntimeError(f'rank {r} failed: {err[0]!r}\n{err[1]}')
    return world.results
'''


# ---------------------------------------------------------------------------
# SPEC 1 — dcgan-normal-init-002 ex2
# ---------------------------------------------------------------------------

SPEC_DCGAN_INIT = {
    "atom_id": "dcgan-normal-init-002",
    "subtopic": "GAN: DCGAN normal init 0.02",
    "topic_folder": TOPIC_DCGAN,
    "atom_recap_md": RECAP_NORMAL_INIT_DEEP,
    "exercise_index": 2,
    "exercise_title": "named-module DCGAN init with per-type gain and a report",
    "slug": "named-module-dcgan-init-with-gain-and-report",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["dcgan", "init", "named-modules", "gain", "report"],
    "kcs": ["named-modules-iter-with-qname", "gain-scaled-normal-init"],
    "lo": (
        "Analyze a model by walking `model.named_modules()`, apply gain-"
        "scaled `N(0, gain*0.02)` init to Conv/ConvTranspose layers (gain "
        "per type), and return a sorted report of `(qualified_name, "
        "type_label, post_init_std)`."
    ),
    "prompt_body": (
        "Implement `ex2_named_dcgan_init(model, conv_gain=1.0, "
        "convt_gain=1.0)`. Three responsibilities:\n\n"
        "1. Walk `model.named_modules()`. SKIP the root entry (qname == "
        "`''`).\n"
        "2. For each yielded `(qname, m)`:\n"
        "   - If `isinstance(m, nn.Conv2d)`: call "
        "`nn.init.normal_(m.weight, 0.0, conv_gain * 0.02)`, then append "
        "`(qname, 'conv2d', m.weight.std().item())` to a `records` list.\n"
        "   - elif `isinstance(m, nn.ConvTranspose2d)`: call "
        "`nn.init.normal_(m.weight, 0.0, convt_gain * 0.02)`, then append "
        "`(qname, 'convtranspose2d', m.weight.std().item())`.\n"
        "   - Other layer types: skip (no init, no record).\n"
        "3. Return `sorted(records)` — sorted by qname (lexicographic, "
        "default tuple sort).\n\n"
        "Input: `model` — `nn.Module`; `conv_gain`, `convt_gain` — floats, "
        "default 1.0.\n"
        "Output: `list[tuple[str, str, float]]`.\n\n"
        "The visualization plots the per-layer post-init std as a bar chart "
        "with two colors (conv vs convt), showing how the gain knobs shift "
        "the std away from the baseline 0.02."
    ),
    "stub": (
        "def ex2_named_dcgan_init(model: nn.Module, conv_gain: float = 1.0, convt_gain: float = 1.0) -> list:\n"
        '    """Init Conv/ConvT via named_modules; return sorted report of (qname, type, std)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "\n"
        "# Build a model with named submodules so qnames are non-trivial.\n"
        "class Block(nn.Module):\n"
        "    def __init__(self, ic, oc):\n"
        "        super().__init__()\n"
        "        self.conv = nn.Conv2d(ic, oc, 3, padding=1)\n"
        "        self.bn = nn.BatchNorm2d(oc)\n"
        "        self.up = nn.ConvTranspose2d(oc, oc, 4, stride=2, padding=1)\n"
        "\n"
        "model = nn.Sequential()\n"
        "model.add_module('block_a', Block(3, 8))\n"
        "model.add_module('block_b', Block(8, 16))\n"
        "model.add_module('head', nn.Linear(16, 10))\n"
        "\n"
        "bn_w_before = model.block_a.bn.weight.detach().clone()\n"
        "lin_w_before = model.head.weight.detach().clone()\n"
        "\n"
        "# Run with default gains (1.0, 1.0) — std should be ~0.02 for both types.\n"
        "report = ex2_named_dcgan_init(model)\n"
        "assert isinstance(report, list), f'expected list, got {type(report).__name__}'\n"
        "assert len(report) == 4, f'expected 4 records (2 conv + 2 convT), got {len(report)}: {report}'\n"
        "\n"
        "# Records sorted by qname.\n"
        "qnames = [r[0] for r in report]\n"
        "assert qnames == sorted(qnames), f'records not sorted by qname: {qnames}'\n"
        "\n"
        "# All expected qnames present.\n"
        "expected_qnames = {'block_a.conv', 'block_a.up', 'block_b.conv', 'block_b.up'}\n"
        "assert set(qnames) == expected_qnames, f'qnames wrong: {set(qnames)} vs {expected_qnames}'\n"
        "\n"
        "# Type labels correct.\n"
        "for qname, tlabel, std in report:\n"
        "    if 'conv' == qname.split('.')[-1]:\n"
        "        assert tlabel == 'conv2d', f'{qname}: expected conv2d, got {tlabel}'\n"
        "    elif 'up' == qname.split('.')[-1]:\n"
        "        assert tlabel == 'convtranspose2d', f'{qname}: expected convtranspose2d, got {tlabel}'\n"
        "    assert abs(std - 0.02) < 0.012, f'{qname}: std {std:.5f} not ~0.02'\n"
        "\n"
        "# BatchNorm + Linear untouched.\n"
        "assert t.equal(model.block_a.bn.weight, bn_w_before), 'BN must be untouched'\n"
        "assert t.equal(model.head.weight, lin_w_before), 'Linear must be untouched'\n"
        "\n"
        "# Gain knobs widen the per-type std.\n"
        "model2 = nn.Sequential()\n"
        "model2.add_module('block_a', Block(3, 8))\n"
        "model2.add_module('block_b', Block(8, 16))\n"
        "report2 = ex2_named_dcgan_init(model2, conv_gain=3.0, convt_gain=5.0)\n"
        "conv_stds = [s for _, tl, s in report2 if tl == 'conv2d']\n"
        "convt_stds = [s for _, tl, s in report2 if tl == 'convtranspose2d']\n"
        "for s in conv_stds:\n"
        "    assert abs(s - 0.06) < 0.025, f'conv std with gain=3 should be ~0.06, got {s:.5f}'\n"
        "for s in convt_stds:\n"
        "    assert abs(s - 0.10) < 0.04, f'convT std with gain=5 should be ~0.10, got {s:.5f}'\n"
        "\n"
        "# Root entry must be skipped — no record with qname == ''.\n"
        "assert all(r[0] != '' for r in report), 'root qname must be filtered out'\n"
        "\n"
        "# Empty model (no Conv/ConvT) → empty report.\n"
        "bare = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))\n"
        "assert ex2_named_dcgan_init(bare) == [], 'model with no conv layers should give empty report'\n"
        "\n"
        "# --- Visualization: per-layer std bar chart with type colors ---\n"
        "viz = nn.Sequential()\n"
        "viz.add_module('block_a', Block(3, 16))\n"
        "viz.add_module('block_b', Block(16, 32))\n"
        "viz.add_module('block_c', Block(32, 64))\n"
        "report_viz = ex2_named_dcgan_init(viz, conv_gain=2.0, convt_gain=4.0)\n"
        "colors = ['steelblue' if tl == 'conv2d' else 'coral' for _, tl, _ in report_viz]\n"
        "labels = [q for q, _, _ in report_viz]\n"
        "vals = [s for _, _, s in report_viz]\n"
        "fig, ax = plt.subplots(figsize=(9, 4))\n"
        "ax.bar(range(len(vals)), vals, color=colors, edgecolor='black')\n"
        "ax.axhline(0.02, ls='--', color='gray', label='baseline std=0.02')\n"
        "ax.axhline(0.04, ls=':', color='steelblue', label='conv (gain=2) → 0.04')\n"
        "ax.axhline(0.08, ls=':', color='coral', label='convT (gain=4) → 0.08')\n"
        "ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha='right')\n"
        "ax.set_ylabel('post-init std'); ax.set_title('DCGAN named-module init — per-type gain')\n"
        "ax.legend(); plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex2_named_dcgan_init(model: nn.Module, conv_gain: float = 1.0, convt_gain: float = 1.0) -> list:\n"
        "    records = []\n"
        "    for qname, m in model.named_modules():\n"
        "        if qname == '':\n"
        "            continue\n"
        "        if isinstance(m, nn.Conv2d):\n"
        "            nn.init.normal_(m.weight, 0.0, conv_gain * 0.02)\n"
        "            records.append((qname, 'conv2d', m.weight.std().item()))\n"
        "        elif isinstance(m, nn.ConvTranspose2d):\n"
        "            nn.init.normal_(m.weight, 0.0, convt_gain * 0.02)\n"
        "            records.append((qname, 'convtranspose2d', m.weight.std().item()))\n"
        "    return sorted(records)"
    ),
    "solution_notes": (
        "**`named_modules()` over `apply`.** `apply` gives you the module "
        "but not its name — fine for pure init, bad for logging. The named "
        "form is the canonical pattern in training scripts that emit "
        "per-layer stats to wandb or tensorboard.\n\n"
        "**Filter root via `qname == ''`.** The first item from "
        "`named_modules()` is always the root model — empty qname, model "
        "itself as the module. Forgetting to skip it doesn't break the "
        "init (the model itself isn't a Conv2d) but it appears as a stray "
        "in any qname-keyed dict.\n\n"
        "**Gain as a multiplier, not a replacement.** Multiplying 0.02 by "
        "a gain factor lets you keep the DCGAN base std while tuning per "
        "type. Common in production: `gain=1.4` on the final ConvT to "
        "compensate for the Tanh activation."
    ),
    "extra_imports": [
        "import torch.nn as nn",
        "import matplotlib.pyplot as plt",
    ],
}


# ---------------------------------------------------------------------------
# SPEC 2 — generator-loss-fool-discriminator ex2 (logits form)
# ---------------------------------------------------------------------------

SPEC_G_LOSS = {
    "atom_id": "generator-loss-fool-discriminator",
    "subtopic": "GAN: Generator loss to fool D",
    "topic_folder": TOPIC_DCGAN,
    "atom_recap_md": RECAP_G_LOSS_LOGITS,
    "exercise_index": 2,
    "exercise_title": "logits-form non-saturating G loss via BCE-with-logits",
    "slug": "generator-bce-with-logits",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["gan", "generator-loss", "bce-logits", "numerical-stability"],
    "kcs": ["bce-with-logits-form", "logit-vs-probability-domain"],
    "lo": (
        "Apply `F.binary_cross_entropy_with_logits` with `t.ones_like` "
        "targets to compute the numerically-stable generator loss on D's "
        "logits (pre-sigmoid)."
    ),
    "prompt_body": (
        "Implement `ex2_generator_loss_from_logits(d_logits)`. The "
        "numerically-stable form of ex1's G loss — now consuming D's "
        "logits directly:\n\n"
        "1. `d_logits` is D's pre-sigmoid output on the fake batch, shape "
        "`(B,)`, values in `(-inf, +inf)`.\n"
        "2. Build targets: `targets = t.ones_like(d_logits)`.\n"
        "3. Return `F.binary_cross_entropy_with_logits(d_logits, targets)` "
        "— a scalar.\n\n"
        "Why this version. `binary_cross_entropy_with_logits` fuses sigmoid "
        "+ BCE and uses `log1p(exp(-z))` internally. It stays finite for "
        "`|z| < ~80`, whereas the bare `binary_cross_entropy(sigmoid(z), "
        "1)` blows up at `|z| > ~16` due to log(0).\n\n"
        "Input: `d_logits` — `(B,)` float tensor.\n"
        "Output: scalar tensor.\n\n"
        "The visualization plots G loss as a function of the logit value, "
        "with the bare-prob form overlaid — showing the numerical "
        "divergence at large positive/negative logits."
    ),
    "stub": (
        "def ex2_generator_loss_from_logits(d_logits: Tensor) -> Tensor:\n"
        '    """Non-saturating G loss in logits domain: BCE-with-logits(d_logits, ones)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn.functional as F\n"
        "import math\n"
        "\n"
        "# Logit 0 → sigmoid=0.5 → BCE_with_logits(0, 1) = log(2) ≈ 0.6931.\n"
        "z_zero = t.zeros(8)\n"
        "loss_zero = ex2_generator_loss_from_logits(z_zero)\n"
        "assert loss_zero.dim() == 0, 'loss must be scalar'\n"
        "expected_zero = math.log(2)\n"
        "assert abs(loss_zero.item() - expected_zero) < 1e-5, f'logit=0 expected {expected_zero:.5f}, got {loss_zero.item():.5f}'\n"
        "\n"
        "# Very positive logit → D fully fooled → G loss → 0.\n"
        "z_pos = t.full((8,), 10.0)\n"
        "loss_pos = ex2_generator_loss_from_logits(z_pos)\n"
        "assert loss_pos.item() < 1e-4, f'large positive logit should give tiny loss, got {loss_pos.item():.6f}'\n"
        "\n"
        "# Very negative logit → D not fooled → G loss → large (≈ -z for big negative).\n"
        "z_neg = t.full((8,), -10.0)\n"
        "loss_neg = ex2_generator_loss_from_logits(z_neg)\n"
        "assert abs(loss_neg.item() - 10.0) < 0.01, f'logit=-10 should give loss ~10, got {loss_neg.item():.4f}'\n"
        "\n"
        "# EXTREME stability test — at logit=-80, bare prob form would crash with log(0).\n"
        "# With-logits form must stay finite.\n"
        "z_extreme = t.full((4,), -80.0)\n"
        "loss_extreme = ex2_generator_loss_from_logits(z_extreme)\n"
        "assert t.isfinite(loss_extreme).item(), f'must be finite at extreme logit, got {loss_extreme.item()}'\n"
        "assert abs(loss_extreme.item() - 80.0) < 0.1, f'logit=-80 should give loss ~80, got {loss_extreme.item()}'\n"
        "\n"
        "# Numerical match against the reference.\n"
        "t.manual_seed(0)\n"
        "logits = t.randn(16) * 3.0\n"
        "got = ex2_generator_loss_from_logits(logits)\n"
        "expected = F.binary_cross_entropy_with_logits(logits, t.ones_like(logits))\n"
        "assert t.allclose(got, expected, atol=1e-6), f'numerical mismatch: {got.item()} vs {expected.item()}'\n"
        "\n"
        "# Gradient: d/dz BCE_with_logits(z, 1) = sigmoid(z) - 1 < 0, so grad on z is negative.\n"
        "z_g = t.full((4,), 0.0, requires_grad=True)\n"
        "ex2_generator_loss_from_logits(z_g).backward()\n"
        "assert (z_g.grad < 0).all(), 'gradient should push logit UP (toward fooled-D)'\n"
        "# At z=0, grad magnitude = sigmoid(0) - 1 = -0.5, averaged over batch → -0.5/B.\n"
        "expected_grad = -0.5 / 4\n"
        "assert t.allclose(z_g.grad, t.full((4,), expected_grad), atol=1e-6), f'grad expected ~{expected_grad}, got {z_g.grad}'\n"
        "\n"
        "# --- Visualization: G loss vs logit, plus bare-prob form for comparison ---\n"
        "zs = t.linspace(-12.0, 12.0, 200)\n"
        "losses_logits = [ex2_generator_loss_from_logits(t.full((4,), z.item())).item() for z in zs]\n"
        "# bare-prob form: sigmoid then BCE on probabilities — clip to avoid log(0).\n"
        "probs = t.sigmoid(zs).clamp(1e-7, 1 - 1e-7)\n"
        "losses_probs = [-math.log(p.item()) for p in probs]   # = BCE(p, 1)\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "ax.plot(zs.numpy(), losses_logits, label='BCE-with-logits (stable)', color='seagreen', lw=2)\n"
        "ax.plot(zs.numpy(), losses_probs, label='sigmoid→BCE-prob (clipped)', color='coral', lw=2, ls='--')\n"
        "ax.set_xlabel('D logit on fakes'); ax.set_ylabel('generator loss')\n"
        "ax.set_title('Non-saturating G loss: logits form matches bare-prob form, stays finite further out')\n"
        "ax.legend(); ax.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex2_generator_loss_from_logits(d_logits: Tensor) -> Tensor:\n"
        "    import torch.nn.functional as F\n"
        "    targets = t.ones_like(d_logits)\n"
        "    return F.binary_cross_entropy_with_logits(d_logits, targets)"
    ),
    "solution_notes": (
        "**Why the logit form is the production default.** In ex1, "
        "`d_pred = sigmoid(linear_output)` then `BCE(d_pred, 1)`. Two "
        "places to lose precision: `sigmoid` flushes large negatives to 0, "
        "then `log(0) = -inf`. Fusing the two sidesteps the intermediate "
        "and uses `log1p(exp(-z))` — finite for any `|z| < ~80`.\n\n"
        "**Discriminator output should be a logit, not a probability.** "
        "Drop the `nn.Sigmoid()` from D's last layer when you switch to "
        "`_with_logits`. Forgetting and sigmoid-ing twice is a silent "
        "bug: D's effective output is `sigmoid(sigmoid(x))`, training "
        "works but slower.\n\n"
        "**Same target=1 trick.** The G→fool-D semantic is unchanged. "
        "Only the loss function and the meaning of its first arg shift."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
}


# ---------------------------------------------------------------------------
# SPEC 3 — model-train-eval-toggle-around-sample ex2 (contextmanager)
# ---------------------------------------------------------------------------

SPEC_TRAIN_EVAL_CTX = {
    "atom_id": "model-train-eval-toggle-around-sample",
    "subtopic": "GAN: model.train/eval toggle around sample",
    "topic_folder": TOPIC_DCGAN,
    "atom_recap_md": RECAP_TRAIN_EVAL_CTX,
    "exercise_index": 2,
    "exercise_title": "@contextmanager eval_mode: exception-safe eval/no_grad toggle",
    "slug": "contextmanager-eval-mode-exception-safe",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["contextmanager", "eval-mode", "no_grad", "try-finally", "exception-safe"],
    "kcs": ["contextmanager-eval-restore", "preserve-prior-training-state"],
    "lo": (
        "Apply `@contextmanager` plus `try/finally` to wrap "
        "`model.eval()` + `torch.no_grad()` into a reusable block that "
        "guarantees the prior `model.training` state is restored even if "
        "the inner block raises."
    ),
    "prompt_body": (
        "Implement `ex2_eval_mode(model)`, a context manager. Required "
        "behavior:\n\n"
        "1. Decorate with `@contextlib.contextmanager`.\n"
        "2. Capture `was_training = model.training` BEFORE entering eval.\n"
        "3. Call `model.eval()`.\n"
        "4. Open `with t.no_grad():` and `yield model` from inside it.\n"
        "5. Wrap the whole `yield` in `try/finally`. In `finally`: if "
        "`was_training` was True, call `model.train()`. Otherwise (model "
        "was already in eval), leave it in eval.\n\n"
        "Critical: the `finally` block must run even if the body of the "
        "`with` raises — that's the entire point of the wrapper.\n\n"
        "Input: `model` — `nn.Module`.\n"
        "Yields: `model` (still the same instance, now in eval + no_grad "
        "context).\n\n"
        "The visualization runs a normal `with ex2_eval_mode(model):` "
        "block, then a raising one, and plots BN's `running_mean` state "
        "to confirm it's untouched both ways."
    ),
    "stub": (
        "import contextlib\n"
        "\n"
        "@contextlib.contextmanager\n"
        "def ex2_eval_mode(model: nn.Module):\n"
        '    """Eval + no_grad context manager with exception-safe restore."""\n'
        "    raise NotImplementedError()\n"
        "    yield model  # unreachable — keeps generator framing"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "\n"
        "# Build a model with BN to verify running stats unchanged.\n"
        "model = nn.Sequential(\n"
        "    nn.ConvTranspose2d(10, 32, 4, stride=1, padding=0, bias=False),\n"
        "    nn.BatchNorm2d(32),\n"
        "    nn.ReLU(),\n"
        ")\n"
        "model.train()\n"
        "for _ in range(3):\n"
        "    _ = model(t.randn(8, 10, 1, 1))\n"
        "bn = model[1]\n"
        "rmean_before = bn.running_mean.detach().clone()\n"
        "\n"
        "# Normal use — body runs fine.\n"
        "assert model.training is True, 'precondition: model in train mode'\n"
        "with ex2_eval_mode(model) as m:\n"
        "    assert m is model, 'context manager must yield the same model instance'\n"
        "    assert not m.training, 'inside context, model must be in eval'\n"
        "    assert not bn.training, 'BN submodule must also be in eval'\n"
        "    out = m(10.0 * t.randn(4, 10, 1, 1))\n"
        "    assert not out.requires_grad, 'inside no_grad, output must not require grad'\n"
        "assert model.training is True, 'after normal exit, model must be back in train'\n"
        "assert t.allclose(bn.running_mean, rmean_before), 'BN running_mean must not have changed'\n"
        "\n"
        "# Exception inside the body — `finally` must still restore train mode.\n"
        "class _BoomError(RuntimeError):\n"
        "    pass\n"
        "\n"
        "raised = False\n"
        "try:\n"
        "    with ex2_eval_mode(model) as m:\n"
        "        assert not m.training, 'inside block: eval'\n"
        "        raise _BoomError('forward exploded')\n"
        "except _BoomError:\n"
        "    raised = True\n"
        "assert raised, 'exception must propagate out of the context manager'\n"
        "assert model.training is True, 'after exception, model MUST be restored to train mode'\n"
        "\n"
        "# Nested-call footgun — outer was already eval, inner should leave eval as-is on exit.\n"
        "model.eval()\n"
        "assert model.training is False\n"
        "with ex2_eval_mode(model) as m:\n"
        "    assert not m.training, 'inside nested context: still eval'\n"
        "assert model.training is False, 'outer was eval; must NOT flip to train on inner exit'\n"
        "\n"
        "# Restore for next assertion.\n"
        "model.train()\n"
        "\n"
        "# Same noise twice through the context manager — reproducible (eval mode).\n"
        "rng = t.Generator().manual_seed(0)\n"
        "noise = t.randn(2, 10, 1, 1, generator=rng)\n"
        "with ex2_eval_mode(model) as m:\n"
        "    s1 = m(noise)\n"
        "with ex2_eval_mode(model) as m:\n"
        "    s2 = m(noise)\n"
        "assert t.allclose(s1, s2), 'two eval-mode calls on same noise must match'\n"
        "\n"
        "# --- Visualization: BN running_mean unchanged across multiple context-managed calls ---\n"
        "trace_before = bn.running_mean.detach().clone()\n"
        "for _ in range(10):\n"
        "    with ex2_eval_mode(model) as m:\n"
        "        _ = m(t.randn(4, 10, 1, 1) * 5.0)   # weird-distribution noise\n"
        "trace_after = bn.running_mean.detach().clone()\n"
        "fig, ax = plt.subplots(figsize=(8, 3))\n"
        "ax.plot(trace_before.numpy(), 'o-', color='steelblue', label='before 10 eval-mode samples')\n"
        "ax.plot(trace_after.numpy(), 'x--', color='coral', label='after (should overlay)')\n"
        "ax.set_xlabel('BN channel'); ax.set_ylabel('running_mean')\n"
        "ax.set_title('Context-manager eval_mode: running_mean preserved across 10 sample calls')\n"
        "ax.legend(); ax.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "import contextlib\n"
        "\n"
        "@contextlib.contextmanager\n"
        "def ex2_eval_mode(model: nn.Module):\n"
        "    was_training = model.training\n"
        "    model.eval()\n"
        "    try:\n"
        "        with t.no_grad():\n"
        "            yield model\n"
        "    finally:\n"
        "        if was_training:\n"
        "            model.train()"
    ),
    "solution_notes": (
        "**Pattern: `@contextmanager` + `try/finally`.** The generator-"
        "based form of context-manager construction. Code before `yield` "
        "runs on `__enter__`; code after `yield` runs on `__exit__`. "
        "`try/finally` around the yield is mandatory to guarantee cleanup "
        "on exception.\n\n"
        "**Why preserve `was_training`.** A caller might already be inside "
        "`model.eval()` (e.g. a validation epoch wrapping a metric "
        "evaluator). Flipping unconditionally to train on exit would "
        "corrupt the outer scope's invariant. Capture-and-restore is the "
        "library-grade form.\n\n"
        "**`with t.no_grad():` inside, not as a separate context.** "
        "Stacking `with eval_mode(m), t.no_grad():` would also work — but "
        "embedding no_grad inside the eval_mode wrapper is the single-"
        "context API the rest of the training code consumes."
    ),
    "extra_imports": [
        "import torch.nn as nn",
        "import matplotlib.pyplot as plt",
    ],
}


# ---------------------------------------------------------------------------
# SPEC 4 — module-modules-iter-isinstance-dispatch ex2 (named report)
# ---------------------------------------------------------------------------

SPEC_MODULES_NAMED = {
    "atom_id": "module-modules-iter-isinstance-dispatch",
    "subtopic": "GAN: model.modules() isinstance dispatch",
    "topic_folder": TOPIC_DCGAN,
    "atom_recap_md": RECAP_MODULES_NAMED,
    "exercise_index": 2,
    "exercise_title": "named_modules dispatch — qualified-name to layer-type-label dict",
    "slug": "named-modules-dispatch-name-to-type",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["named-modules", "isinstance", "dispatch", "qualified-name"],
    "kcs": ["named-modules-iter-qname", "skip-root-empty-qname"],
    "lo": (
        "Analyze a model via `model.named_modules()`, dispatching on "
        "isinstance to build a `dict[qualified_name, type_label]` that "
        "skips the root entry and labels the standard DCGAN layer types."
    ),
    "prompt_body": (
        "Implement `ex2_named_layer_report(model)`. Walk every submodule "
        "by qualified name, dispatch by type:\n\n"
        "1. Build an empty dict `report`.\n"
        "2. Iterate `for qname, m in model.named_modules()`. SKIP the "
        "root entry (`qname == ''`).\n"
        "3. Dispatch with `if / elif` (mutually exclusive):\n"
        "   - `nn.Conv2d` → `report[qname] = 'conv2d'`\n"
        "   - `nn.ConvTranspose2d` → `report[qname] = 'convtranspose2d'`\n"
        "   - `nn.BatchNorm2d` or `nn.BatchNorm1d` → `report[qname] = "
        "'batchnorm'`\n"
        "   - `nn.Linear` → `report[qname] = 'linear'`\n"
        "   - Other → SKIP (do not add to report; the report is "
        "type-filtered, not type-complete).\n"
        "4. Return `report`.\n\n"
        "Key differences from ex1:\n"
        "- ex1 returned a `dict[type_label, count]`; ex2 returns a "
        "`dict[qname, type_label]` (no 'other' bucket).\n"
        "- Root is SKIPPED — `'' -> 'sequential'` would be noise.\n"
        "- Activations / Flatten / Sequential / ReLU don't appear.\n\n"
        "Input: `model` — `nn.Module`.\n"
        "Output: `dict[str, str]`.\n\n"
        "The visualization renders the report as a categorical scatter "
        "(x=insertion order, y=type-label, point label = qname)."
    ),
    "stub": (
        "def ex2_named_layer_report(model: nn.Module) -> dict:\n"
        '    """dict[qname, type_label] of conv2d / convtranspose2d / batchnorm / linear submodules."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "\n"
        "# Build a DCGAN-shaped generator with a named encoder/decoder split.\n"
        "class Encoder(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)\n"
        "        self.bn1 = nn.BatchNorm2d(16)\n"
        "        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)\n"
        "        self.bn2 = nn.BatchNorm2d(32)\n"
        "\n"
        "class Decoder(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.up1 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)\n"
        "        self.bn = nn.BatchNorm2d(16)\n"
        "        self.up2 = nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1)\n"
        "        self.flat = nn.Flatten()\n"
        "        self.head = nn.Linear(3, 10)\n"
        "\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.enc = Encoder()\n"
        "        self.dec = Decoder()\n"
        "\n"
        "model = Model()\n"
        "report = ex2_named_layer_report(model)\n"
        "assert isinstance(report, dict), f'expected dict, got {type(report).__name__}'\n"
        "\n"
        "# Required entries — every Conv/ConvT/BN/Linear submodule.\n"
        "expected = {\n"
        "    'enc.conv1': 'conv2d',\n"
        "    'enc.bn1': 'batchnorm',\n"
        "    'enc.conv2': 'conv2d',\n"
        "    'enc.bn2': 'batchnorm',\n"
        "    'dec.up1': 'convtranspose2d',\n"
        "    'dec.bn': 'batchnorm',\n"
        "    'dec.up2': 'convtranspose2d',\n"
        "    'dec.head': 'linear',\n"
        "}\n"
        "assert report == expected, f'report mismatch:\\n got  {sorted(report.items())}\\n want {sorted(expected.items())}'\n"
        "\n"
        "# Root must be skipped.\n"
        "assert '' not in report, 'root qname must be filtered out'\n"
        "\n"
        "# Non-targeted layer types must NOT appear.\n"
        "for qname in report:\n"
        "    assert 'flat' not in qname.split('.')[-1], f'Flatten leaked into report: {qname}'\n"
        "    # Containers (enc, dec) also must be skipped.\n"
        "assert 'enc' not in report and 'dec' not in report, 'container submodules must be skipped'\n"
        "\n"
        "# Empty model → empty report (only root, which is skipped).\n"
        "assert ex2_named_layer_report(nn.Sequential()) == {}\n"
        "\n"
        "# Single BN at the root → report has exactly one entry — but BN module IS the root\n"
        "# (qname=''), so it's filtered. Wrap in Sequential to give it a qname.\n"
        "wrapped = nn.Sequential(nn.BatchNorm2d(4))\n"
        "wrap_rep = ex2_named_layer_report(wrapped)\n"
        "assert wrap_rep == {'0': 'batchnorm'}, f'expected {{0: batchnorm}}, got {wrap_rep}'\n"
        "\n"
        "# Nested Sequential — dotted qnames preserved.\n"
        "nested = nn.Sequential(\n"
        "    nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8)),\n"
        "    nn.ConvTranspose2d(8, 4, 4),\n"
        ")\n"
        "nr = ex2_named_layer_report(nested)\n"
        "assert nr == {'0.0': 'conv2d', '0.1': 'batchnorm', '1': 'convtranspose2d'}, f'nested wrong: {nr}'\n"
        "\n"
        "# --- Visualization: scatter of qname insertion order by type ---\n"
        "items = list(report.items())\n"
        "type_to_y = {'conv2d': 0, 'convtranspose2d': 1, 'batchnorm': 2, 'linear': 3}\n"
        "color_map = {'conv2d': 'steelblue', 'convtranspose2d': 'coral', 'batchnorm': 'seagreen', 'linear': 'gold'}\n"
        "fig, ax = plt.subplots(figsize=(10, 4))\n"
        "for i, (qname, tlabel) in enumerate(items):\n"
        "    ax.scatter(i, type_to_y[tlabel], s=200, color=color_map[tlabel], edgecolor='black', zorder=3)\n"
        "    ax.text(i, type_to_y[tlabel] + 0.15, qname, ha='center', fontsize=8, rotation=20)\n"
        "ax.set_yticks(list(type_to_y.values())); ax.set_yticklabels(list(type_to_y.keys()))\n"
        "ax.set_xlabel('insertion order'); ax.set_xlim(-0.5, len(items) - 0.5)\n"
        "ax.set_title('Named-module report: qname → type'); ax.grid(True, axis='y', alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex2_named_layer_report(model: nn.Module) -> dict:\n"
        "    report = {}\n"
        "    for qname, m in model.named_modules():\n"
        "        if qname == '':\n"
        "            continue\n"
        "        if isinstance(m, nn.Conv2d):\n"
        "            report[qname] = 'conv2d'\n"
        "        elif isinstance(m, nn.ConvTranspose2d):\n"
        "            report[qname] = 'convtranspose2d'\n"
        "        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):\n"
        "            report[qname] = 'batchnorm'\n"
        "        elif isinstance(m, nn.Linear):\n"
        "            report[qname] = 'linear'\n"
        "    return report"
    ),
    "solution_notes": (
        "**Why filter the root.** `named_modules()` yields the root as "
        "`('', model)`. The model itself usually isn't one of the four "
        "target types, but even if it were (e.g. a bare `nn.Conv2d`), the "
        "empty qname is meaningless for instrumentation. Filter on "
        "`qname == ''` rather than relying on the type-check happening to "
        "skip it.\n\n"
        "**Containers (Sequential, Module subclasses) are naturally "
        "filtered.** They're not Conv2d / ConvTranspose2d / BatchNorm / "
        "Linear, so they fall through the dispatch without an `else` "
        "branch — the report is type-filtered by construction.\n\n"
        "**Why `if/elif`, not stacked `if`.** Mutually exclusive — every "
        "qname maps to exactly one type-label. Stacked `if` would allow a "
        "future subclass that satisfies two type checks to silently "
        "overwrite its earlier label."
    ),
    "extra_imports": [
        "import torch.nn as nn",
        "import matplotlib.pyplot as plt",
    ],
}


# ---------------------------------------------------------------------------
# SPEC 5 — noise-batch-from-latent ex2 (device + sphere normalize)
# ---------------------------------------------------------------------------

SPEC_NOISE = {
    "atom_id": "noise-batch-from-latent",
    "subtopic": "GAN: Noise batch from latent_dim",
    "topic_folder": TOPIC_DCGAN,
    "atom_recap_md": RECAP_NOISE_SPHERE,
    "exercise_index": 2,
    "exercise_title": "device-targeted noise with optional unit-sphere normalization",
    "slug": "noise-batch-device-and-sphere-normalize",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["dcgan", "noise", "device", "sphere-normalize", "L2-norm"],
    "kcs": ["randn-with-device-kwarg", "per-sample-l2-normalize-to-unit-sphere"],
    "lo": (
        "Apply `t.randn(B, L, 1, 1, device=device, generator=g)` followed "
        "by optional per-sample L2 normalization (with `clamp_min(1e-8)` "
        "safety) to project latent codes onto the unit hypersphere."
    ),
    "prompt_body": (
        "Implement `ex2_dcgan_noise(batch_size, latent_dim, generator, "
        "device, normalize=False)`. Two-mode noise builder:\n\n"
        "1. Build standard-normal noise of shape `(batch_size, latent_dim, "
        "1, 1)` directly on `device` (use the `device=` kwarg to "
        "`t.randn`, NOT a post-hoc `.to(device)`).\n"
        "2. Pass the `generator` kwarg through for reproducibility.\n"
        "3. If `normalize=True`:\n"
        "   - Compute per-sample L2 norms over the last 3 dims (latent + "
        "spatial): `norms = noise.flatten(1).norm(dim=1)` — shape `(B,)`.\n"
        "   - Divide each sample by its norm, clamping the divisor below "
        "by `1e-8` for safety: "
        "`noise = noise / norms.view(B, 1, 1, 1).clamp_min(1e-8)`.\n"
        "4. Return the (possibly normalized) noise tensor.\n\n"
        "Important shape contract:\n"
        "- Output shape ALWAYS `(B, L, 1, 1)`, with or without normalize.\n"
        "- Output device ALWAYS `device`.\n"
        "- When `normalize=True`, each sample's L2 norm is `1.0 ± 1e-5`.\n\n"
        "Input: `batch_size`, `latent_dim` — ints; `generator` — `t."
        "Generator`; `device` — `t.device` or device-string; `normalize` "
        "— bool, default False.\n"
        "Output: `(B, L, 1, 1)` float32 tensor on `device`.\n\n"
        "The visualization plots per-sample L2 norm histograms for both "
        "modes side by side — the unnormalized form is `χ`-distributed "
        "around √latent_dim, the normalized form is a delta at 1.0."
    ),
    "stub": (
        "def ex2_dcgan_noise(batch_size: int, latent_dim: int, generator: 'torch.Generator',\n"
        "                    device, normalize: bool = False) -> Tensor:\n"
        '    """(B, L, 1, 1) noise on `device`. Optional per-sample unit-sphere normalize."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Shape + device — unnormalized.\n"
        "cpu = t.device('cpu')\n"
        "rng = t.Generator().manual_seed(0)\n"
        "noise = ex2_dcgan_noise(8, 100, rng, cpu, normalize=False)\n"
        "assert noise.shape == (8, 100, 1, 1), f'shape wrong: {tuple(noise.shape)}'\n"
        "assert noise.device == cpu, f'device wrong: {noise.device}'\n"
        "assert noise.dtype == t.float32\n"
        "\n"
        "# Unnormalized — std should be ~1 (standard normal).\n"
        "rng2 = t.Generator().manual_seed(0)\n"
        "big = ex2_dcgan_noise(1000, 100, rng2, cpu, normalize=False)\n"
        "assert abs(big.std().item() - 1.0) < 0.05, f'unnormalized std should be ~1, got {big.std().item():.4f}'\n"
        "assert abs(big.mean().item()) < 0.05, f'unnormalized mean should be ~0, got {big.mean().item():.4f}'\n"
        "\n"
        "# Normalized — every sample's L2 norm exactly 1.\n"
        "rng3 = t.Generator().manual_seed(0)\n"
        "norm_noise = ex2_dcgan_noise(32, 64, rng3, cpu, normalize=True)\n"
        "assert norm_noise.shape == (32, 64, 1, 1), 'shape must be preserved by normalize'\n"
        "per_sample_norms = norm_noise.flatten(1).norm(dim=1)\n"
        "assert per_sample_norms.shape == (32,)\n"
        "assert t.allclose(per_sample_norms, t.ones(32), atol=1e-5), (\n"
        "    f'per-sample norms must all be 1, got mean={per_sample_norms.mean().item():.6f}, '\n"
        "    f'std={per_sample_norms.std().item():.6f}'\n"
        ")\n"
        "\n"
        "# Reproducibility — same seed gives same noise (both modes).\n"
        "for mode in [False, True]:\n"
        "    rng_a = t.Generator().manual_seed(42)\n"
        "    rng_b = t.Generator().manual_seed(42)\n"
        "    a = ex2_dcgan_noise(4, 8, rng_a, cpu, normalize=mode)\n"
        "    b = ex2_dcgan_noise(4, 8, rng_b, cpu, normalize=mode)\n"
        "    assert t.equal(a, b), f'normalize={mode}: same seed must give same noise'\n"
        "\n"
        "# Direction preserved — normalizing should not change the unit-vector direction.\n"
        "rng4 = t.Generator().manual_seed(7)\n"
        "raw = ex2_dcgan_noise(4, 32, rng4, cpu, normalize=False)\n"
        "rng4b = t.Generator().manual_seed(7)\n"
        "normd = ex2_dcgan_noise(4, 32, rng4b, cpu, normalize=True)\n"
        "for i in range(4):\n"
        "    raw_dir = raw[i].flatten() / raw[i].flatten().norm()\n"
        "    norm_dir = normd[i].flatten() / normd[i].flatten().norm()\n"
        "    assert t.allclose(raw_dir, norm_dir, atol=1e-5), f'sample {i}: direction changed by normalize'\n"
        "\n"
        "# Safety: clamp_min in divisor — verify no inf/nan even when very low L makes some norms tiny.\n"
        "# (Construct a degenerate case by post-zeroing one sample.)\n"
        "rng5 = t.Generator().manual_seed(0)\n"
        "test = ex2_dcgan_noise(2, 4, rng5, cpu, normalize=False)\n"
        "# can't easily force a zero sample through randn; trust the clamp by checking small-L behavior.\n"
        "rng6 = t.Generator().manual_seed(0)\n"
        "tiny = ex2_dcgan_noise(64, 2, rng6, cpu, normalize=True)\n"
        "assert t.isfinite(tiny).all(), 'normalize with small L must stay finite (clamp_min safety)'\n"
        "\n"
        "# Input shape passes through a DCGAN first layer.\n"
        "import torch.nn as nn\n"
        "first = nn.ConvTranspose2d(100, 512, 4, stride=1, padding=0, bias=False)\n"
        "rng7 = t.Generator().manual_seed(0)\n"
        "n = ex2_dcgan_noise(2, 100, rng7, cpu, normalize=True)\n"
        "out = first(n)\n"
        "assert out.shape == (2, 512, 4, 4), f'first-layer output wrong: {tuple(out.shape)}'\n"
        "\n"
        "# --- Visualization: per-sample L2 norm histogram, unnormalized vs normalized ---\n"
        "rng_v = t.Generator().manual_seed(0)\n"
        "raw_viz = ex2_dcgan_noise(2000, 100, rng_v, cpu, normalize=False)\n"
        "rng_v2 = t.Generator().manual_seed(0)\n"
        "norm_viz = ex2_dcgan_noise(2000, 100, rng_v2, cpu, normalize=True)\n"
        "raw_norms = raw_viz.flatten(1).norm(dim=1).numpy()\n"
        "norm_norms = norm_viz.flatten(1).norm(dim=1).numpy()\n"
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))\n"
        "ax1.hist(raw_norms, bins=50, color='steelblue', edgecolor='black')\n"
        "ax1.axvline(100 ** 0.5, color='red', ls='--', label=f'expected √{100}={100**0.5:.2f}')\n"
        "ax1.set_title('unnormalized — χ-like distribution around √L'); ax1.legend()\n"
        "ax2.hist(norm_norms, bins=10, range=(0.9, 1.1), color='coral', edgecolor='black')\n"
        "ax2.set_xlim(0.9, 1.1)\n"
        "ax2.set_title('normalized — delta at 1.0 (unit sphere)')\n"
        "for ax in (ax1, ax2):\n"
        "    ax.set_xlabel('per-sample L2 norm'); ax.set_ylabel('count')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex2_dcgan_noise(batch_size: int, latent_dim: int, generator: 'torch.Generator',\n"
        "                    device, normalize: bool = False) -> Tensor:\n"
        "    noise = t.randn(batch_size, latent_dim, 1, 1, device=device, generator=generator)\n"
        "    if normalize:\n"
        "        norms = noise.flatten(1).norm(dim=1)\n"
        "        noise = noise / norms.view(batch_size, 1, 1, 1).clamp_min(1e-8)\n"
        "    return noise"
    ),
    "solution_notes": (
        "**`device=` in `t.randn` avoids a host→device copy.** Building on "
        "CPU then `.to('cuda')` does two allocations and one PCIe copy. "
        "Passing `device=` builds the tensor on the target device directly "
        "— a meaningful speedup when sampling thousands of batches in a "
        "real training loop.\n\n"
        "**Why per-sample (not whole-batch) normalization.** Each sample "
        "is an independent latent code; the unit sphere lives in "
        "`latent_dim`-space. Normalizing the entire batch tensor as one "
        "vector would couple samples together — nonsense.\n\n"
        "**`flatten(1)` then `norm(dim=1)`.** Collapses the L + spatial "
        "dims into a single feature axis per sample, then computes L2 "
        "along it. Returns shape `(B,)`. The `view(B, 1, 1, 1)` "
        "broadcasts the per-sample scalar back across the original shape.\n\n"
        "**Why `clamp_min(1e-8)`, not `clamp(min=1e-8)`.** Same call — "
        "`clamp_min` is the one-arg form. Either works; `clamp_min` reads "
        "tighter."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
}


# ---------------------------------------------------------------------------
# SPEC 6 — all-reduce-compose ex2 (MAX op composition)
# ---------------------------------------------------------------------------

SPEC_ALL_REDUCE_MAX = {
    "atom_id": "all-reduce-compose",
    "subtopic": "Distributed: all_reduce composition",
    "topic_folder": TOPIC_DIST,
    "atom_recap_md": RECAP_ALL_REDUCE_MAX,
    "exercise_index": 2,
    "exercise_title": "compose all_reduce with MAX op via reduce + broadcast",
    "slug": "compose-all-reduce-max-via-reduce-broadcast",
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["all_reduce", "MAX", "reduce", "broadcast", "composition"],
    "kcs": ["all-reduce-max-via-reduce-broadcast", "reduce-op-parametric"],
    "lo": (
        "Apply the `reduce(MAX) → broadcast` composition to build a custom "
        "all-reduce that gives every rank the global maximum across "
        "rank-local tensors."
    ),
    "prompt_body": (
        "Implement `ex2_all_reduce_max(rank, world_size, dist_module, "
        "local_value)`. The same `reduce + broadcast` composition shape "
        "as ex1, but with the MAX op:\n\n"
        "1. Build a 1-D tensor wrapping the rank-local value: `tensor = "
        "t.tensor([local_value], dtype=t.float32)`.\n"
        "2. Reduce to rank 0 with the MAX op: `dist_module.reduce(tensor, "
        "dst=0, op=dist_module.ReduceOp.MAX)`. After this, only rank 0's "
        "tensor holds the global maximum.\n"
        "3. Broadcast from rank 0 so every rank sees it: "
        "`dist_module.broadcast(tensor, src=0)`.\n"
        "4. Return `tensor.item()` — the global maximum, identical on "
        "every rank.\n\n"
        "Note the signature change vs ex1: this drill takes `dist_module` "
        "as a parameter (injected by the test harness — a real `dist` on "
        "GPU machines, a mock on CPU). Inside the function, ALL calls go "
        "via `dist_module.*` instead of the global `dist.*`. This is a "
        "common testing pattern — dependency injection makes the function "
        "verifiable on CPU.\n\n"
        "Input: `rank`, `world_size` — ints; `dist_module` — the "
        "torch.distributed module (or a mock); `local_value` — float.\n"
        "Output: `float` — the global max, same on every rank.\n\n"
        "The test simulates `world_size` ranks via threads and a fake "
        "`dist` module that implements `reduce` / `broadcast` / `ReduceOp` "
        "with thread-barrier synchronization."
    ),
    "stub": (
        "def ex2_all_reduce_max(rank: int, world_size: int, dist_module, local_value: float) -> float:\n"
        '    """Compose all_reduce(MAX) from reduce(MAX) + broadcast; return global max."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        _FAKE_DIST_HARNESS + "\n\n"
        "# Per-rank local values: rank r has value (1 + r) * 2.5 → [2.5, 5.0, 7.5, 10.0].\n"
        "# Global MAX → 10.0 on every rank.\n"
        "def _worker(rank, world_size, dist_module, world):\n"
        "    local_value = (rank + 1) * 2.5\n"
        "    result = ex2_all_reduce_max(rank, world_size, dist_module, local_value)\n"
        "    world.results[rank] = result\n"
        "\n"
        "results = _run_fake_world(_worker, 4)\n"
        "expected_max = 10.0\n"
        "for rank, r in enumerate(results):\n"
        "    assert r is not None, f'rank {rank} returned None — function did not complete'\n"
        "    assert abs(r - expected_max) < 1e-5, f'rank {rank}: got {r}, expected {expected_max}'\n"
        "\n"
        "# Negative values — MAX is sign-aware.\n"
        "def _worker_neg(rank, world_size, dist_module, world):\n"
        "    local_value = -float(rank + 1)   # [-1, -2, -3]\n"
        "    world.results[rank] = ex2_all_reduce_max(rank, world_size, dist_module, local_value)\n"
        "\n"
        "results_neg = _run_fake_world(_worker_neg, 3)\n"
        "expected_neg_max = -1.0   # the LEAST negative\n"
        "for rank, r in enumerate(results_neg):\n"
        "    assert abs(r - expected_neg_max) < 1e-5, f'neg: rank {rank}: got {r}, expected {expected_neg_max}'\n"
        "\n"
        "# Same value on every rank — global max is that value.\n"
        "def _worker_same(rank, world_size, dist_module, world):\n"
        "    world.results[rank] = ex2_all_reduce_max(rank, world_size, dist_module, 42.0)\n"
        "\n"
        "results_same = _run_fake_world(_worker_same, 5)\n"
        "for rank, r in enumerate(results_same):\n"
        "    assert abs(r - 42.0) < 1e-5, f'identical-values case rank {rank}: got {r}'\n"
        "\n"
        "# Single-rank world — degenerate, max = own value.\n"
        "def _worker_single(rank, world_size, dist_module, world):\n"
        "    world.results[rank] = ex2_all_reduce_max(rank, world_size, dist_module, 7.5)\n"
        "\n"
        "results_single = _run_fake_world(_worker_single, 1)\n"
        "assert abs(results_single[0] - 7.5) < 1e-5"
    ),
    "solution_body": (
        "def ex2_all_reduce_max(rank: int, world_size: int, dist_module, local_value: float) -> float:\n"
        "    tensor = t.tensor([local_value], dtype=t.float32)\n"
        "    dist_module.reduce(tensor, dst=0, op=dist_module.ReduceOp.MAX)\n"
        "    dist_module.broadcast(tensor, src=0)\n"
        "    return tensor.item()"
    ),
    "solution_notes": (
        "**The composition is `op`-parametric.** Same `reduce → broadcast` "
        "shape for SUM, MAX, MIN, PRODUCT — only the `op=` kwarg changes. "
        "This is why ARENA teaches the composition: once you know it, "
        "every `all_reduce_*` variant slots in.\n\n"
        "**Why `dist_module` as a parameter.** Tests on CPU can't run "
        "real `gloo`/`nccl` (no fork on Windows, no GPUs on Colab CPU "
        "runtimes). Injecting the dist module lets the test pass in a "
        "mock that simulates `world_size` ranks via threads + barriers. "
        "Production code re-binds `dist_module = torch.distributed` at "
        "the call site.\n\n"
        "**`reduce(MAX)` + `broadcast` vs real `all_reduce(MAX)`.** "
        "Functionally identical. Real `all_reduce` uses tree-reduction "
        "(`O(log world_size)` rounds) while compose uses linear (`O(2 * "
        "world_size)`). For learning, compose. For production, `dist."
        "all_reduce` directly."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 7 — all-reduce-eval-metrics ex2 (weighted mean)
# ---------------------------------------------------------------------------

SPEC_EVAL_WEIGHTED = {
    "atom_id": "all-reduce-eval-metrics",
    "subtopic": "Distributed: all_reduce eval metrics",
    "topic_folder": TOPIC_DIST,
    "atom_recap_md": RECAP_ALL_REDUCE_WEIGHTED,
    "exercise_index": 2,
    "exercise_title": "sample-count-weighted eval mean via packed all_reduce",
    "slug": "sample-count-weighted-eval-mean-via-packed-all-reduce",
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["all_reduce", "weighted-mean", "uneven-batch", "packed-tensor"],
    "kcs": ["pack-sum-and-count-into-one-tensor", "weighted-mean-via-sum-divide"],
    "lo": (
        "Apply a packed `(sum_loss, count)` `all_reduce(SUM)` followed by "
        "a single divide to compute the true global mean of an eval "
        "metric across ranks with unequal sample counts."
    ),
    "prompt_body": (
        "Implement `ex2_weighted_eval_mean(rank, world_size, "
        "dist_module, local_loss_sum, local_count)`. The "
        "uneven-batch-correct version of ex1's eval mean:\n\n"
        "1. Pack `(local_loss_sum, local_count)` into a single length-2 "
        "tensor: `stats = t.tensor([local_loss_sum, float(local_count)], "
        "dtype=t.float32)`. (Cast count to float — `all_reduce` requires "
        "a float tensor.)\n"
        "2. Run ONE `all_reduce(SUM)` over the packed tensor. After this, "
        "`stats[0]` is the global sum of losses and `stats[1]` is the "
        "global sample count.\n"
        "3. Compute the true weighted mean: `global_mean = stats[0] / "
        "stats[1]`. (Returned as a Python float via `.item()`.)\n"
        "4. Return `global_mean`.\n\n"
        "Why packed.** Two all_reduces double the network round-trip "
        "cost; one packed all_reduce halves it. Math is identical.\n\n"
        "Why this matters (the bug ex1 hides).** If rank 0 sees 32 "
        "samples with mean loss 1.0 and rank 1 sees 8 samples with mean "
        "loss 5.0:\n"
        "- Naive `mean of means` = (1 + 5) / 2 = 3.0 (WRONG — over-"
        "weights small rank).\n"
        "- Weighted = (32*1 + 8*5) / (32 + 8) = 72/40 = 1.8 (RIGHT — "
        "every sample counted once).\n\n"
        "Input: `rank`, `world_size` — ints; `dist_module` — torch."
        "distributed or mock; `local_loss_sum` — float (sum, not mean!); "
        "`local_count` — int.\n"
        "Output: `float` — true global mean, same on every rank."
    ),
    "stub": (
        "def ex2_weighted_eval_mean(rank: int, world_size: int, dist_module,\n"
        "                           local_loss_sum: float, local_count: int) -> float:\n"
        '    """Weighted global mean via packed (sum, count) all_reduce(SUM) + divide."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        _FAKE_DIST_HARNESS + "\n\n"
        "# The classic uneven-batch case.\n"
        "# Rank 0: 32 samples, mean=1.0 (sum=32). Rank 1: 8 samples, mean=5.0 (sum=40).\n"
        "# Weighted mean = (32 + 40) / (32 + 8) = 72/40 = 1.8.\n"
        "_rank_sums   = [32.0, 40.0]\n"
        "_rank_counts = [32, 8]\n"
        "\n"
        "def _worker(rank, world_size, dist_module, world):\n"
        "    result = ex2_weighted_eval_mean(rank, world_size, dist_module,\n"
        "                                    _rank_sums[rank], _rank_counts[rank])\n"
        "    world.results[rank] = result\n"
        "\n"
        "results = _run_fake_world(_worker, 2)\n"
        "expected = 72.0 / 40.0   # = 1.8\n"
        "for rank, r in enumerate(results):\n"
        "    assert r is not None, f'rank {rank} returned None'\n"
        "    assert abs(r - expected) < 1e-5, (\n"
        "        f'rank {rank}: got {r}, expected {expected}.  '\n"
        "        f'If you got 3.0, you computed mean-of-means (the bug this drills out).'\n"
        "    )\n"
        "\n"
        "# Equal batches — weighted mean degenerates to plain mean.\n"
        "_equal_sums = [10.0, 20.0, 30.0]\n"
        "_equal_counts = [10, 10, 10]\n"
        "\n"
        "def _worker_equal(rank, world_size, dist_module, world):\n"
        "    world.results[rank] = ex2_weighted_eval_mean(\n"
        "        rank, world_size, dist_module, _equal_sums[rank], _equal_counts[rank])\n"
        "\n"
        "results_equal = _run_fake_world(_worker_equal, 3)\n"
        "expected_equal = (10 + 20 + 30) / 30   # = 2.0\n"
        "for rank, r in enumerate(results_equal):\n"
        "    assert abs(r - expected_equal) < 1e-5, f'equal-batch rank {rank}: got {r}'\n"
        "\n"
        "# Zero-count rank — should not crash (assuming at least one rank has count > 0).\n"
        "_zero_sums = [50.0, 0.0, 100.0]\n"
        "_zero_counts = [10, 0, 20]   # rank 1 had no samples\n"
        "\n"
        "def _worker_zero(rank, world_size, dist_module, world):\n"
        "    world.results[rank] = ex2_weighted_eval_mean(\n"
        "        rank, world_size, dist_module, _zero_sums[rank], _zero_counts[rank])\n"
        "\n"
        "results_zero = _run_fake_world(_worker_zero, 3)\n"
        "expected_zero = (50 + 0 + 100) / (10 + 0 + 20)   # = 150/30 = 5.0\n"
        "for rank, r in enumerate(results_zero):\n"
        "    assert abs(r - expected_zero) < 1e-5, f'zero-count case rank {rank}: got {r}'\n"
        "\n"
        "# Floating-point precision test — large counts.\n"
        "_large_sums = [1000.5, 2000.5, 3000.5, 4000.5]\n"
        "_large_counts = [100, 200, 300, 400]\n"
        "\n"
        "def _worker_large(rank, world_size, dist_module, world):\n"
        "    world.results[rank] = ex2_weighted_eval_mean(\n"
        "        rank, world_size, dist_module, _large_sums[rank], _large_counts[rank])\n"
        "\n"
        "results_large = _run_fake_world(_worker_large, 4)\n"
        "expected_large = sum(_large_sums) / sum(_large_counts)\n"
        "for rank, r in enumerate(results_large):\n"
        "    assert abs(r - expected_large) < 1e-4, f'large-batch rank {rank}: got {r}, expected {expected_large}'"
    ),
    "solution_body": (
        "def ex2_weighted_eval_mean(rank: int, world_size: int, dist_module,\n"
        "                           local_loss_sum: float, local_count: int) -> float:\n"
        "    stats = t.tensor([local_loss_sum, float(local_count)], dtype=t.float32)\n"
        "    dist_module.all_reduce(stats, op=dist_module.ReduceOp.SUM)\n"
        "    global_mean = stats[0] / stats[1]\n"
        "    return global_mean.item()"
    ),
    "solution_notes": (
        "**Why pack into one tensor.** Each `all_reduce` is a network "
        "round-trip; on a fast interconnect (NVLink, IB) each costs "
        "microseconds, but they add up across thousands of eval batches "
        "in a long training run. Packing two scalars into one tensor "
        "halves the all_reduce count.\n\n"
        "**`local_loss_sum`, not `local_mean`.** This is the subtle bug "
        "fix vs ex1. If you reduce per-rank MEANS, the count weight is "
        "lost; the weighted-mean math no longer works. Pass the "
        "unreduced sum (or sum of loss × batch_size, depending on how "
        "you computed local loss).\n\n"
        "**Float cast on count.** `all_reduce` requires a float tensor on "
        "gloo (and integer reduce is finicky on NCCL too). Cast `count` "
        "to `float32` on the way in; cast back if you really need an int "
        "(usually you don't — divide stays float).\n\n"
        "**Division by zero edge case.** If EVERY rank has count 0 "
        "(rare), `stats[1] = 0` and division gives `inf`/`nan`. In "
        "practice the eval pipeline guards against this upstream; we "
        "trust the caller."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 8 — all-reduce-grad-sync ex2 (skip-sync for None grads)
# ---------------------------------------------------------------------------

SPEC_GRAD_SKIP = {
    "atom_id": "all-reduce-grad-sync",
    "subtopic": "Distributed: all_reduce grad sync",
    "topic_folder": TOPIC_DIST,
    "atom_recap_md": RECAP_GRAD_SYNC_SKIP,
    "exercise_index": 2,
    "exercise_title": "skip-sync optimization — only all_reduce non-None grads",
    "slug": "skip-sync-optimization-only-all-reduce-non-none-grads",
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["DDP", "grad-sync", "skip-sync", "frozen-layers", "sparse-grads"],
    "kcs": ["skip-none-grads-in-sync-loop", "all-reduce-mean-divide-after-skip"],
    "lo": (
        "Apply the `if p.grad is None: continue` skip-sync optimization "
        "before `dist.all_reduce(p.grad, SUM)` + divide-by-world_size, "
        "verified by running on a model with one frozen layer."
    ),
    "prompt_body": (
        "Implement `ex2_grad_sync_skip_none(rank, world_size, "
        "dist_module, model)`. The frozen-layer-aware grad sync:\n\n"
        "1. Loop `for p in model.parameters()`.\n"
        "2. **If `p.grad is None`: `continue`.** This skips parameters "
        "that didn't receive a gradient on this step (frozen layers, "
        "sparse heads, etc.).\n"
        "3. Otherwise, all-reduce the grad: `dist_module.all_reduce("
        "p.grad, op=dist_module.ReduceOp.SUM)`.\n"
        "4. Divide by `world_size` in-place: `p.grad /= world_size`.\n"
        "5. Return the integer COUNT of parameters that were synced (i.e. "
        "had non-None grads). The test asserts this count matches "
        "expectations.\n\n"
        "Important: every rank runs this same loop on the same model "
        "graph, so the `is None` decision is consistent across ranks — "
        "no risk of deadlock.\n\n"
        "Input: `rank`, `world_size` — ints; `dist_module` — torch."
        "distributed or mock; `model` — `nn.Module` with some "
        "parameters' `.grad` set, some left as `None`.\n"
        "Output: `int` — count of parameters synced (i.e. that had a "
        "non-None grad)."
    ),
    "stub": (
        "def ex2_grad_sync_skip_none(rank: int, world_size: int, dist_module, model: 'nn.Module') -> int:\n"
        '    """Sync grads across ranks, skipping params with grad is None. Return synced count."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        _FAKE_DIST_HARNESS + "\n\n"
        "import torch.nn as nn\n"
        "\n"
        "# Model with two params; rank-specific grads.\n"
        "# Param `active` always gets a grad. Param `frozen` is left at grad=None.\n"
        "class _TwoParam(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.active = nn.Parameter(t.zeros(3))\n"
        "        self.frozen = nn.Parameter(t.zeros(3))\n"
        "\n"
        "def _worker(rank, world_size, dist_module, world):\n"
        "    model = _TwoParam()\n"
        "    # Each rank gets a DIFFERENT grad on `active`.\n"
        "    model.active.grad = t.tensor([float(rank + 1)] * 3)\n"
        "    # `frozen.grad` stays None — simulates frozen layer / sparse head.\n"
        "    assert model.frozen.grad is None, 'precondition: frozen.grad starts None'\n"
        "    synced = ex2_grad_sync_skip_none(rank, world_size, dist_module, model)\n"
        "    world.results[rank] = (synced, model.active.grad.tolist(), model.frozen.grad)\n"
        "\n"
        "# 3 ranks: active.grad = [1,1,1], [2,2,2], [3,3,3] → mean [2,2,2]. frozen stays None.\n"
        "results = _run_fake_world(_worker, 3)\n"
        "for rank, res in enumerate(results):\n"
        "    assert res is not None, f'rank {rank} returned None'\n"
        "    synced, active_grad, frozen_grad = res\n"
        "    assert synced == 1, f'rank {rank}: expected 1 param synced (only active), got {synced}'\n"
        "    expected = [2.0, 2.0, 2.0]\n"
        "    for i, (a, b) in enumerate(zip(active_grad, expected)):\n"
        "        assert abs(a - b) < 1e-5, f'rank {rank} active.grad[{i}]: got {a}, expected {b}'\n"
        "    assert frozen_grad is None, f'rank {rank}: frozen.grad must remain None (was skipped), got {frozen_grad}'\n"
        "\n"
        "# 2 ranks, all params have grad — synced count = 2.\n"
        "def _worker_all_grads(rank, world_size, dist_module, world):\n"
        "    model = _TwoParam()\n"
        "    model.active.grad = t.tensor([float(rank + 1)] * 3)\n"
        "    model.frozen.grad = t.tensor([float((rank + 1) * 10)] * 3)\n"
        "    synced = ex2_grad_sync_skip_none(rank, world_size, dist_module, model)\n"
        "    world.results[rank] = (synced, model.active.grad.tolist(), model.frozen.grad.tolist())\n"
        "\n"
        "results_all = _run_fake_world(_worker_all_grads, 2)\n"
        "for rank, res in enumerate(results_all):\n"
        "    synced, active_grad, frozen_grad = res\n"
        "    assert synced == 2, f'rank {rank}: expected 2 params synced, got {synced}'\n"
        "    # active: [1,1,1] and [2,2,2] → mean [1.5,1.5,1.5]\n"
        "    for v in active_grad:\n"
        "        assert abs(v - 1.5) < 1e-5\n"
        "    # frozen: [10,10,10] and [20,20,20] → mean [15,15,15]\n"
        "    for v in frozen_grad:\n"
        "        assert abs(v - 15.0) < 1e-5\n"
        "\n"
        "# All params have grad None → synced count = 0, no all_reduce called, no crash.\n"
        "def _worker_no_grads(rank, world_size, dist_module, world):\n"
        "    model = _TwoParam()\n"
        "    # Both stay None.\n"
        "    synced = ex2_grad_sync_skip_none(rank, world_size, dist_module, model)\n"
        "    world.results[rank] = (synced, model.active.grad, model.frozen.grad)\n"
        "\n"
        "results_none = _run_fake_world(_worker_no_grads, 3)\n"
        "for rank, res in enumerate(results_none):\n"
        "    synced, a, f = res\n"
        "    assert synced == 0, f'rank {rank}: expected 0 synced, got {synced}'\n"
        "    assert a is None and f is None, 'grads must stay None when skipped'"
    ),
    "solution_body": (
        "def ex2_grad_sync_skip_none(rank: int, world_size: int, dist_module, model: 'nn.Module') -> int:\n"
        "    synced = 0\n"
        "    for p in model.parameters():\n"
        "        if p.grad is None:\n"
        "            continue\n"
        "        dist_module.all_reduce(p.grad, op=dist_module.ReduceOp.SUM)\n"
        "        p.grad /= world_size\n"
        "        synced += 1\n"
        "    return synced"
    ),
    "solution_notes": (
        "**Why `is None`, not `== 0`.** A zero-VALUED grad tensor still "
        "needs to be reduced — it's just one rank's contribution that "
        "happens to be zero. `None` means 'no grad was computed this "
        "step' (e.g. backward was never called for this param's "
        "subgraph). Only the latter should be skipped.\n\n"
        "**Rank consistency is load-bearing.** If rank 0 thinks `p.grad "
        "is None` and rank 1 thinks it has a grad, rank 1 calls "
        "`all_reduce` with no counterpart — deadlock. In practice every "
        "rank runs the same forward/backward on the same model graph, so "
        "the `is None` answer is identical. The invariant is preserved "
        "by the data-parallelism contract, not by any check in this "
        "function.\n\n"
        "**Real DDP fuses + skips automatically.** "
        "`torch.nn.parallel.DistributedDataParallel` hooks backward and "
        "kicks an all_reduce when each param's grad becomes ready — "
        "skipping frozen layers naturally. The hand-rolled loop you're "
        "writing is the conceptual model; DDP is the optimized "
        "production version."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# All specs
# ---------------------------------------------------------------------------

SPECS = [
    SPEC_DCGAN_INIT,
    SPEC_G_LOSS,
    SPEC_TRAIN_EVAL_CTX,
    SPEC_MODULES_NAMED,
    SPEC_NOISE,
    SPEC_ALL_REDUCE_MAX,
    SPEC_EVAL_WEIGHTED,
    SPEC_GRAD_SKIP,
]


# ---------------------------------------------------------------------------
# Verifier — exec stub + solution + test_body inside a single namespace.
# ---------------------------------------------------------------------------

def _verify_all(specs):
    import torch as t
    import numpy as np
    import torch.nn as nn
    import matplotlib
    matplotlib.use("Agg")   # headless — no display in verifier
    import matplotlib.pyplot as plt
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
            "plt": plt,
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
            plt.close("all")
            continue
        plt.close("all")
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
    print(f"[deepening_q_batch10] Verifying {len(SPECS)} specs...")
    _verify_all(SPECS)

    print(f"\n[deepening_q_batch10] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_q_batch10] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
