#!/usr/bin/env python3
"""Author 8 deepening (ex2) drills across generative + geometry-cnn atoms.

Each ex2 hits a DISTINCT facet from the corresponding ex1 — different cognitive
operation, surface context, or scale. ONE LO + ONE Bloom + <=2 KCs per drill.

Verification re-runs each spec's solution against its test_body inside the
build venv (torch 2.12.0+cpu) before any notebook is emitted.

Atom roster (8 atoms, all ex2):
  prereqs_generative/
    mse-reconstruction-loss              — ex2: per-pixel saliency map
    randn-like-noise-source              — ex2: fix the float32 leak + dtype audit
    requires-grad-leaf-assert            — ex2: spot the .to(device) silent-no-op
    t-stack-trajectory                   — ex2: window-and-diff a trajectory
    two-optimizers-alternating-step      — ex2: WGAN-style k D-steps per G-step
  prereqs_geometry_cnn/
    cross-product-normal                 — ex2: batched normals + degenerate mask
    rotation-matrix-3d                   — ex2: compose two Rodrigues rotations
    segment-line-intersect-2d            — ex2: batched segments vs a single line
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC_GENERATIVE = "prereqs_generative"
TOPIC_GEOMETRY_CNN = "prereqs_geometry_cnn"


# ---------------------------------------------------------------------------
# Per-atom recap blocks (trimmed for deepening focus).
# ---------------------------------------------------------------------------

RECAP_MSE = (
    "## MSE reconstruction loss — deepening refresher\n"
    "\n"
    "`F.mse_loss(decoded, original)` has three reductions:\n"
    "- `'mean'` (default) → scalar averaged over **every** element\n"
    "- `'sum'` → scalar summed over every element\n"
    "- `'none'` → per-element tensor with the original shape, you choose how "
    "to collapse it\n"
    "\n"
    "**Per-axis averaging.** Once you have `(B, 1, H, W)` per-element error, "
    "you can average over any subset of axes. Averaging over `[0, 1]` gives "
    "a `(H, W)` **saliency map** — the average per-pixel reconstruction error "
    "across the whole batch. Hot spots reveal pixels the autoencoder "
    "struggles with regardless of which sample it sees (often the edges of "
    "digits, or the corners that are usually zero).\n"
    "\n"
    "**Why this is NOT what you backprop.** The saliency map is for "
    "DIAGNOSIS — the scalar `.mean()` loss is what `.backward()` consumes. "
    "Backpropagating a `(H, W)` tensor implicitly sums it, which is the "
    "wrong reduction (training would be `H*W` times stronger)."
)

RECAP_RANDN_LIKE = (
    "## `randn_like` noise source — deepening refresher\n"
    "\n"
    "`t.randn_like(x)` produces standard-normal noise with the SAME shape, "
    "dtype, and device as `x`. The two-arg cousin `t.randn(*x.shape)` matches "
    "shape only — dtype defaults to `float32`, device defaults to CPU.\n"
    "\n"
    "**The silent dtype leak.** If your encoder runs in `float64` (or mixed "
    "precision in `float16`), the `randn(*shape)` form returns `float32` "
    "noise. Then `mu + sigma * eps` either upcasts the whole pipeline to "
    "`float32` (silently losing precision) or — worse — raises a confusing "
    "type-promotion error deep in the loss.\n"
    "\n"
    "**The audit pattern.** When inheriting a buggy reparameterization "
    "function, assert dtype propagation on the OUTPUT:\n"
    "```python\n"
    "z = reparam(mu, sigma)\n"
    "assert z.dtype == sigma.dtype, f'reparam leaked dtype: {z.dtype} vs {sigma.dtype}'\n"
    "```\n"
    "If that assertion ever fires, the call site is using `randn(*shape)` "
    "instead of `randn_like`."
)

RECAP_LEAF_ASSERT = (
    "## `requires_grad` + leaf assertion — deepening refresher\n"
    "\n"
    "An optimizer updates LEAF tensors with `requires_grad=True`. A leaf "
    "tensor has no `.grad_fn` — it's either an `nn.Parameter` or a tensor "
    "you explicitly constructed with `requires_grad=True`. Op outputs are "
    "non-leaves; `opt.step()` silently skips them.\n"
    "\n"
    "**The classic silent bug.** You call `.to('cpu')` (or `.to(device)`) "
    "on an `nn.Parameter` to move it. The return value is a NEW non-leaf "
    "tensor — `.to()` on a tensor that requires grad treats the move as an "
    "op, so the result has a `grad_fn`. You hand the result to "
    "`Adam([...], lr=...)`, the optimizer accepts it, runs `.step()` without "
    "errors, and updates NOTHING. Training loss plateaus, you debug the model "
    "for a day. The fix: move the whole `nn.Module` (`model.to(device)`) — "
    "Module's `.to()` rebinds parameters in place.\n"
    "\n"
    "**The diagnostic pattern.** Given a list of would-be optimizer params, "
    "audit each one and return a STRUCTURED report — which params are leaf "
    "(safe), which are not (silent skip), which have `requires_grad=False` "
    "(also silent skip)."
)

RECAP_STACK_TRAJ = (
    "## `torch.stack` trajectory — deepening refresher\n"
    "\n"
    "A trajectory `(B, T, D)` is the canonical layout from stacking a Python "
    "list of `(B, D)` per-step latents with `t.stack(..., dim=1)`. Once you "
    "have the trajectory, three operations dominate downstream analysis:\n"
    "\n"
    "1. **Window slicing.** `traj[:, t0:t1, :]` extracts a contiguous time "
    "window. Shape becomes `(B, t1 - t0, D)`.\n"
    "2. **Per-step deltas.** `delta = traj[:, 1:, :] - traj[:, :-1, :]` gives "
    "the per-step changes. Shape collapses from `T` to `T - 1` along the "
    "time axis — every other axis is preserved.\n"
    "3. **Reductions over time.** `traj.mean(dim=1)` averages along the time "
    "axis to give a `(B, D)` summary. `traj.std(dim=1)` gives per-dim "
    "volatility.\n"
    "\n"
    "**Why `dim=1` everywhere.** Once you've committed to `(B, T, D)` (the "
    "standard sequence convention), time slicing, diffing, and reducing are "
    "all `dim=1` operations. Mixing `dim=0` operations on a `(B, T, D)` "
    "tensor accidentally walks the batch axis — usually visible only as "
    "a weird-looking learning curve."
)

RECAP_TWO_OPTIMIZERS = (
    "## GAN: two-optimizers alternating step — deepening refresher\n"
    "\n"
    "The vanilla pattern is **one D-step + one G-step per iteration**. But "
    "in practice the discriminator often needs MORE updates than the "
    "generator to stay strong enough to produce useful gradients (WGAN-GP "
    "uses `n_critic = 5` D-steps per G-step).\n"
    "\n"
    "```python\n"
    "for iteration in range(N):\n"
    "    # k discriminator updates\n"
    "    for _ in range(n_critic):\n"
    "        D_opt.zero_grad()\n"
    "        z = torch.randn(B, z_dim)\n"
    "        fake = G(z).detach()             # stop-grad into G\n"
    "        loss_D = (D(fake) - D(x_real)).mean()\n"
    "        loss_D.backward()\n"
    "        D_opt.step()\n"
    "    # 1 generator update\n"
    "    G_opt.zero_grad()\n"
    "    z = torch.randn(B, z_dim)\n"
    "    fake = G(z)                          # grad flows into G\n"
    "    loss_G = -D(fake).mean()\n"
    "    loss_G.backward()\n"
    "    G_opt.step()\n"
    "```\n"
    "\n"
    "**Invariants.** Each D-step uses a FRESH `z` (you can also reuse — both "
    "are seen in the wild). The G-step uses its own fresh `z`, NOT detached. "
    "Every step zeros only ITS optimizer's grads. `n_critic=1` recovers the "
    "Goodfellow alternation."
)

RECAP_CROSS_NORMAL = (
    "## Cross-product surface normal — deepening refresher\n"
    "\n"
    "For a single triangle, `n = cross(P2-P1, P3-P1)` then normalize. For a "
    "BATCH of `(N, 3, 3)` triangles (N triangles, each with 3 vertices, each "
    "vertex 3-D), the same recipe applies along the last axis:\n"
    "\n"
    "```python\n"
    "e1 = tris[:, 1] - tris[:, 0]          # (N, 3)\n"
    "e2 = tris[:, 2] - tris[:, 0]          # (N, 3)\n"
    "n = t.linalg.cross(e1, e2, dim=-1)     # (N, 3) — un-normalized\n"
    "norms = n.norm(dim=-1, keepdim=True)   # (N, 1)\n"
    "```\n"
    "\n"
    "**Degeneracy.** A triangle is degenerate iff its three vertices are "
    "colinear ⇒ `cross == 0` ⇒ `||n|| == 0` ⇒ division by zero produces "
    "`nan` / `inf`. Real renderers test `norms > eps` (eps ~ 1e-8) and "
    "either skip the triangle or substitute a sentinel normal.\n"
    "\n"
    "**Returning a mask is better than skipping.** A boolean `valid: (N,)` "
    "lets downstream code decide what to do (skip in lighting, but maybe "
    "keep for connectivity)."
)

RECAP_ROT3D = (
    "## 3-D rotation (Rodrigues) — deepening refresher\n"
    "\n"
    "Rodrigues builds the rotation matrix `R = I + sinθ·K + (1-cosθ)·K²` "
    "from a unit axis `k` and angle `θ`. Composition of two rotations is "
    "MATRIX MULTIPLICATION:\n"
    "\n"
    "```\n"
    "R_total = R2 @ R1     # apply R1 first, then R2\n"
    "```\n"
    "\n"
    "**Non-commutativity.** `R1 @ R2 != R2 @ R1` in general — rotations "
    "around different axes don't commute. This is why robotics IK and "
    "camera control distinguish 'apply pitch then yaw' from 'apply yaw "
    "then pitch'.\n"
    "\n"
    "**Closure.** The composition of two rotation matrices is ITSELF a "
    "rotation matrix. Numerical proof: `R_total @ R_total.T ≈ I` and "
    "`det(R_total) ≈ +1`. The set of 3×3 rotations forms a group (SO(3)) "
    "under matrix multiplication.\n"
    "\n"
    "**Why this matters for ARENA.** Composing rotations is how you build "
    "a camera transform (model-space → world-space → camera-space) or "
    "chain joints in a kinematic tree."
)

RECAP_SEG_LINE = (
    "## 2-D segment vs line intersection — deepening refresher\n"
    "\n"
    "A single intersection is a 2×2 linear system. For a BATCH of `N` "
    "segments vs ONE infinite line, broadcast the construction and solve "
    "all `N` systems in a single `t.linalg.solve` call:\n"
    "\n"
    "```python\n"
    "d = S1 - S0                          # (N, 2)  segment directions\n"
    "e = L1 - L0                          # (2,)    one line direction\n"
    "A = t.stack([d, -e.expand_as(d)], dim=-1)   # (N, 2, 2): columns d and -e\n"
    "b = L0 - S0                          # (N, 2)\n"
    "ts = t.linalg.solve(A, b)            # (N, 2)  — [t_i, s_i] per segment\n"
    "```\n"
    "\n"
    "**Hit mask.** `hit = (ts[..., 0] >= 0) & (ts[..., 0] <= 1)` — closed "
    "interval on segment parameter, line parameter unconstrained.\n"
    "\n"
    "**Parallel-segment handling.** When a segment is parallel to the line, "
    "the corresponding 2×2 is singular and `linalg.solve` raises. For a "
    "batched call, the entire batch raises even if just one segment is "
    "parallel. The robust workaround is to mask via determinant before "
    "solving (covered separately in `singular-matrix-mask-trick`)."
)


SPECS = [
    # ===================================================================
    # mse-reconstruction-loss  —  ex2
    # ex1 = scalar + per-sample. NEW facet: per-pixel saliency map across
    # the batch (reduction='none', then average over [batch, channel] axes
    # to expose a (H, W) error heatmap). Bloom: Analyze.
    # ===================================================================
    {
        "atom_id": "mse-reconstruction-loss",
        "subtopic": "Generative: MSE reconstruction loss",
        "topic_folder": TOPIC_GENERATIVE,
        "atom_recap_md": RECAP_MSE,
        "exercise_index": 2,
        "exercise_title": "per-pixel saliency map from per-element MSE",
        "slug": "per-pixel-saliency-map-from-per-element-mse",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["mse", "reduction-none", "saliency", "diagnosis"],
        "kcs": [
            "mse-reduction-none-shape",
            "mean-over-batch-and-channel-axes",
        ],
        "lo": (
            "Analyze a batch of MNIST reconstructions by computing the "
            "(H, W) average per-pixel MSE across the batch and channel "
            "axes, identifying where the autoencoder fails most."
        ),
        "prompt_body": (
            "Implement `ex2_pixel_saliency(original, decoded)`. Build the "
            "average per-pixel reconstruction error across a whole batch — "
            "a diagnostic heatmap that tells you which pixels the "
            "autoencoder gets wrong regardless of which image it sees.\n\n"
            "1. `original` and `decoded` both have shape `(B, 1, H, W)`.\n"
            "2. Call `F.mse_loss(decoded, original, reduction='none')` — "
            "result has shape `(B, 1, H, W)` (per-element squared error).\n"
            "3. Average over the batch axis (`dim=0`) AND the channel axis "
            "(`dim=1`) to collapse to `(H, W)`. Use a single `.mean(dim=[0, 1])` "
            "call.\n"
            "4. Return the `(H, W)` saliency map.\n\n"
            "**Do NOT** call `F.mse_loss(..., reduction='mean')` first and "
            "then try to recover per-pixel detail — once you've reduced to "
            "a scalar, the per-pixel info is gone.\n\n"
            "**Do NOT** use a Python loop over the batch. The whole point is "
            "a single tensor reduction.\n\n"
            "Inputs: `(B, 1, H, W)` float tensors.\n"
            "Output: `(H, W)` float tensor — the average per-pixel error."
        ),
        "stub": (
            "def ex2_pixel_saliency(original: Tensor, decoded: Tensor) -> Tensor:\n"
            '    """Return (H, W) saliency = mean per-element MSE across batch+channel."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import torch.nn.functional as F\n"
            "\n"
            "# Identity case → saliency is all-zero.\n"
            "rng = t.Generator().manual_seed(0)\n"
            "orig = t.rand(8, 1, 7, 7, generator=rng)\n"
            "sal0 = ex2_pixel_saliency(orig, orig.clone())\n"
            "assert sal0.shape == (7, 7), f'expected (H=7, W=7), got {tuple(sal0.shape)}'\n"
            "assert t.allclose(sal0, t.zeros(7, 7), atol=1e-7), (\n"
            "    f'identity saliency must be all-zero, got max={sal0.max().item():.6f}'\n"
            ")\n"
            "\n"
            "# Hand-built case: only one pixel ever differs.\n"
            "# decoded equals original except pixel (3, 4) is shifted by +1 on EVERY batch element.\n"
            "orig2 = t.zeros(5, 1, 7, 7)\n"
            "dec2 = orig2.clone()\n"
            "dec2[:, 0, 3, 4] = 1.0\n"
            "sal = ex2_pixel_saliency(orig2, dec2)\n"
            "assert sal.shape == (7, 7), f'(H,W) shape, got {tuple(sal.shape)}'\n"
            "# Only (3,4) should be nonzero; it should equal (1-0)^2 = 1.0 (mean over batch=5 with all-1).\n"
            "expected_one = t.zeros(7, 7); expected_one[3, 4] = 1.0\n"
            "assert t.allclose(sal, expected_one, atol=1e-6), (\n"
            "    f'expected single hot pixel at (3,4) with value 1.0, got nonzero at '\n"
            "    f'{(sal > 1e-6).nonzero().tolist()}, max value {sal.max().item():.6f}'\n"
            ")\n"
            "\n"
            "# Cross-check against ground truth via F.mse_loss reduction='none'.\n"
            "orig3 = t.rand(6, 1, 7, 7, generator=t.Generator().manual_seed(1))\n"
            "dec3  = t.rand(6, 1, 7, 7, generator=t.Generator().manual_seed(2))\n"
            "got  = ex2_pixel_saliency(orig3, dec3)\n"
            "gt   = F.mse_loss(dec3, orig3, reduction='none').mean(dim=[0, 1])\n"
            "assert got.shape == (7, 7)\n"
            "assert t.allclose(got, gt, atol=1e-6), (\n"
            "    f'saliency disagrees with F.mse_loss(reduction=none).mean([0,1])\\n'\n"
            "    f'max abs diff = {(got - gt).abs().max().item():.6e}'\n"
            ")\n"
            "\n"
            "# Per-pixel asymmetry: corrupt only LEFT half of every image.\n"
            "orig4 = t.zeros(3, 1, 4, 6)\n"
            "dec4 = orig4.clone()\n"
            "dec4[:, 0, :, :3] = 0.5            # left half shifted by 0.5\n"
            "sal4 = ex2_pixel_saliency(orig4, dec4)\n"
            "# Left half: each pixel error = 0.5^2 = 0.25. Right half: 0.\n"
            "assert sal4.shape == (4, 6)\n"
            "assert t.allclose(sal4[:, :3], t.full((4, 3), 0.25), atol=1e-6), (\n"
            "    'left half should have saliency 0.25 everywhere'\n"
            ")\n"
            "assert t.allclose(sal4[:, 3:], t.zeros(4, 3), atol=1e-6), (\n"
            "    'right half should have zero saliency'\n"
            ")"
        ),
        "solution_body": (
            "def ex2_pixel_saliency(original: Tensor, decoded: Tensor) -> Tensor:\n"
            "    import torch.nn.functional as F\n"
            "    per_elem = F.mse_loss(decoded, original, reduction='none')   # (B, 1, H, W)\n"
            "    return per_elem.mean(dim=[0, 1])                              # (H, W)"
        ),
        "solution_notes": (
            "**Why `reduction='none'`.** It's the only reduction that "
            "preserves the `(B, 1, H, W)` shape, leaving you free to "
            "reduce over a chosen subset of axes. `'mean'` and `'sum'` "
            "both collapse to a scalar — you can't recover per-pixel "
            "information after that.\n\n"
            "**Why `dim=[0, 1]` and not `dim=0` then `dim=0` again.** "
            "Calling `mean(dim=0)` twice in a row would average over batch "
            "(`dim=0` of `(B,1,H,W)`) and then again over the channel axis "
            "(now `dim=0` of `(1,H,W)`). Functionally equivalent here "
            "because the channel size is 1, but `dim=[0, 1]` is the "
            "explicit single-call form that scales to multi-channel "
            "images (e.g. `(B, 3, H, W)` for RGB).\n\n"
            "**Saliency vs training loss.** This is a DIAGNOSTIC. The "
            "scalar `F.mse_loss(decoded, original)` is what `.backward()` "
            "consumes — backpropagating a `(H, W)` map would implicitly "
            "sum it and apply `H*W`-times-too-strong gradients."
        ),
        "extra_imports": [],
    },

    # ===================================================================
    # randn-like-noise-source  —  ex2
    # ex1 = apply randn_like for reparam. NEW facet: given a BUGGY function
    # that hardcodes float32 noise, audit + fix it, returning a structured
    # report of (before_dtype, after_dtype, leaked). Bloom: Analyze.
    # ===================================================================
    {
        "atom_id": "randn-like-noise-source",
        "subtopic": "Generative: randn-like noise source",
        "topic_folder": TOPIC_GENERATIVE,
        "atom_recap_md": RECAP_RANDN_LIKE,
        "exercise_index": 2,
        "exercise_title": "audit a reparameterization for dtype leaks",
        "slug": "audit-a-reparameterization-for-dtype-leaks",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["randn_like", "dtype-audit", "debug", "reparameterization"],
        "kcs": [
            "randn-like-vs-randn-shape",
            "audit-output-dtype",
        ],
        "lo": (
            "Analyze a reparameterization function by checking whether its "
            "output dtype matches the input sigma's dtype across float32 / "
            "float64 / float16, returning a structured leak report."
        ),
        "prompt_body": (
            "Two reparam functions are provided in the stub:\n"
            "- `reparam_buggy(mu, sigma)` uses `t.randn(*sigma.shape)` "
            "(hardcodes float32 / CPU).\n"
            "- `reparam_fixed(mu, sigma)` uses `t.randn_like(sigma)` "
            "(inherits dtype + device).\n\n"
            "Implement `ex2_dtype_audit(reparam_fn, dtypes)`. For each "
            "`dtype` in the input list:\n\n"
            "1. Build `mu = t.zeros(8, 4, dtype=dtype)` and "
            "`sigma = t.ones(8, 4, dtype=dtype)`.\n"
            "2. Call `z = reparam_fn(mu, sigma)`.\n"
            "3. Record a triple `(dtype, z.dtype, leaked)` where `leaked` is "
            "`True` iff `z.dtype != dtype`.\n"
            "4. Return a list of those triples — one per input dtype, in order.\n\n"
            "The function MUST be agnostic to which reparam was passed; the "
            "test calls it with both `reparam_buggy` and `reparam_fixed` and "
            "expects the buggy one to leak on `float64` / `float16` while the "
            "fixed one never leaks.\n\n"
            "Input: `reparam_fn` (callable), `dtypes` (list of `torch.dtype`).\n"
            "Output: `list[tuple[torch.dtype, torch.dtype, bool]]`."
        ),
        "stub": (
            "def reparam_buggy(mu: Tensor, sigma: Tensor) -> Tensor:\n"
            "    # BUG: t.randn(*sigma.shape) hardcodes float32 / CPU.\n"
            "    eps = t.randn(*sigma.shape)\n"
            "    return mu + sigma * eps\n"
            "\n"
            "\n"
            "def reparam_fixed(mu: Tensor, sigma: Tensor) -> Tensor:\n"
            "    eps = t.randn_like(sigma)\n"
            "    return mu + sigma * eps\n"
            "\n"
            "\n"
            "def ex2_dtype_audit(reparam_fn, dtypes: list) -> list:\n"
            '    """Audit reparam_fn across dtypes; return list of (in, out, leaked)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "dtypes = [t.float32, t.float64, t.float16]\n"
            "\n"
            "# Buggy reparam: float32 input is fine; float64 and float16 LEAK\n"
            "# (silent upcast or, depending on torch version, an error during\n"
            "# the multiplication). We use a try/except in our audit by virtue\n"
            "# of running the reparam — but if it raises, the leak is obvious.\n"
            "report_buggy = ex2_dtype_audit(reparam_buggy, dtypes)\n"
            "assert isinstance(report_buggy, list), 'must return a list'\n"
            "assert len(report_buggy) == 3, f'expected 3 entries, got {len(report_buggy)}'\n"
            "for entry in report_buggy:\n"
            "    assert isinstance(entry, tuple) and len(entry) == 3, (\n"
            "        f'each entry must be a 3-tuple, got {entry}'\n"
            "    )\n"
            "\n"
            "# float32 → no leak.\n"
            "in_dt, out_dt, leaked = report_buggy[0]\n"
            "assert in_dt == t.float32, f'first entry must be float32, got {in_dt}'\n"
            "assert out_dt == t.float32, f'buggy on float32 should yield float32, got {out_dt}'\n"
            "assert leaked is False, f'float32 input must NOT leak (buggy is harmless here), got {leaked}'\n"
            "\n"
            "# float64 → leak (buggy returns float32 noise; multiplied by float64 sigma promotes\n"
            "# the noise to float64, BUT the issue is the noise itself was generated as float32.\n"
            "# The output of `mu + sigma * eps` ends up float64 because torch promotes UP. So\n"
            "# the LEAK we detect: noise was generated in the wrong dtype, not the output dtype.\n"
            "# Adjust the audit to compare with what randn returns directly, by inspecting via\n"
            "# the helper: we re-run reparam_fn on identical inputs and check NOISE dtype indirectly\n"
            "# by checking whether the output equals mu + sigma * (something cast back to dtype).\n"
            "# Simpler: assert that on float16, the output dtype IS float32 (upcast surprise) since\n"
            "# float16 + float32 promotes to float32.\n"
            "in_dt, out_dt, leaked = report_buggy[2]  # float16\n"
            "assert in_dt == t.float16, f'third entry must be float16, got {in_dt}'\n"
            "# When sigma is float16 and eps is float32, sigma * eps promotes to float32 →\n"
            "# the result dtype is float32, not float16 → LEAKED.\n"
            "assert out_dt == t.float32, (\n"
            "    f'buggy reparam on float16 sigma must produce float32 output (upcast), got {out_dt}'\n"
            ")\n"
            "assert leaked is True, f'float16 input MUST be flagged as leaked, got {leaked}'\n"
            "\n"
            "# Fixed reparam: no leaks anywhere.\n"
            "report_fixed = ex2_dtype_audit(reparam_fixed, dtypes)\n"
            "assert len(report_fixed) == 3\n"
            "for entry in report_fixed:\n"
            "    in_dt, out_dt, leaked = entry\n"
            "    assert in_dt == out_dt, (\n"
            "        f'reparam_fixed leaked dtype: in={in_dt} out={out_dt} — '\n"
            "        'randn_like should preserve dtype'\n"
            "    )\n"
            "    assert leaked is False, f'fixed reparam should never leak, got {entry}'\n"
            "\n"
            "# Order preservation.\n"
            "assert [r[0] for r in report_buggy] == dtypes, 'audit must preserve input order'\n"
            "assert [r[0] for r in report_fixed] == dtypes, 'audit must preserve input order'"
        ),
        "solution_body": (
            "def ex2_dtype_audit(reparam_fn, dtypes):\n"
            "    report = []\n"
            "    for dt in dtypes:\n"
            "        mu = t.zeros(8, 4, dtype=dt)\n"
            "        sigma = t.ones(8, 4, dtype=dt)\n"
            "        z = reparam_fn(mu, sigma)\n"
            "        leaked = (z.dtype != dt)\n"
            "        report.append((dt, z.dtype, leaked))\n"
            "    return report"
        ),
        "solution_notes": (
            "**Why float16 exposes the bug, not float32.** When sigma is "
            "`float32` and noise is `float32` (from `randn(*shape)`), "
            "everything is type-consistent — no leak. When sigma is "
            "`float16`, the multiplication `sigma * eps` promotes the "
            "result to `float32` (the wider of the two), and the output "
            "is silently upcast. Your downstream graph thinks it's "
            "training in `float16` but is actually running in `float32` "
            "for most of the loss path — kills mixed-precision speedup.\n\n"
            "**Why float64 might or might not flag.** When sigma is "
            "`float64` and noise is `float32`, the multiplication promotes "
            "to `float64` — so the output `z.dtype` is `float64`, matching "
            "the input. The leak is INVISIBLE at the output dtype level "
            "(but the noise had less precision than the rest of the "
            "computation). This audit catches the visible-leak case "
            "(`float16` → `float32` upcast); the harder invisible case "
            "needs intermediate-dtype tracking, which is out of scope.\n\n"
            "**The fix is mechanical.** Always use `randn_like(sigma)` "
            "for reparameterization noise — it inherits all three of "
            "(shape, dtype, device) and never leaks."
        ),
        "extra_imports": [],
    },

    # ===================================================================
    # requires-grad-leaf-assert  —  ex2
    # ex1 = assert helper raises on first bad param. NEW facet: a
    # diagnostic that reports ALL bad params (not just the first) and
    # categorizes them. Bloom: Analyze.
    # ===================================================================
    {
        "atom_id": "requires-grad-leaf-assert",
        "subtopic": "Generative: requires_grad leaf assert",
        "topic_folder": TOPIC_GENERATIVE,
        "atom_recap_md": RECAP_LEAF_ASSERT,
        "exercise_index": 2,
        "exercise_title": "categorize bad optimizer params (non-leaf vs no-grad)",
        "slug": "categorize-bad-optimizer-params-non-leaf-vs-no-grad",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["leaf", "requires_grad", "diagnosis", "optimizer-debug"],
        "kcs": [
            "categorize-leaf-vs-non-leaf",
            "report-all-failing-params",
        ],
        "lo": (
            "Analyze a list of would-be optimizer params and produce a "
            "structured report of which indices are non-leaf, which lack "
            "requires_grad, and which are safe to optimize."
        ),
        "prompt_body": (
            "Implement `ex2_audit_optim_params(params)`. Unlike the ex1 "
            "assertion helper which raises on the FIRST bad param, this "
            "deepening exercise produces a full report so the user can see "
            "every problem at once.\n\n"
            "For each tensor `p` at index `i`:\n\n"
            "- If `p.is_leaf is True` AND `p.requires_grad is True` ⇒ the "
            "param is SAFE for the optimizer. Skip it.\n"
            "- Else, append a dict to a `problems` list:\n"
            "  ```python\n"
            "  {\n"
            "      'index': i,\n"
            "      'shape': tuple(p.shape),\n"
            "      'is_leaf': bool(p.is_leaf),\n"
            "      'requires_grad': bool(p.requires_grad),\n"
            "      'category': <'non-leaf' | 'no-grad'>,\n"
            "  }\n"
            "  ```\n"
            "- The `category` field is `'non-leaf'` whenever "
            "`p.is_leaf is False` (regardless of `requires_grad`), and "
            "`'no-grad'` when the param IS a leaf but has "
            "`requires_grad=False`. Non-leaf takes priority because a "
            "non-leaf tensor is the more fundamental break: even if you "
            "fixed `requires_grad`, the optimizer still couldn't update it.\n\n"
            "Return the `problems` list. If every param is safe, return `[]`. "
            "Do NOT raise.\n\n"
            "Input: list of `Tensor` (possibly `nn.Parameter`).\n"
            "Output: `list[dict]` — empty iff every param is optimizer-safe."
        ),
        "stub": (
            "def ex2_audit_optim_params(params) -> list:\n"
            '    """Return a list of problem dicts (one per bad param). Empty list = all safe."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch import nn\n"
            "\n"
            "# All-safe case: every param is a fresh nn.Parameter.\n"
            "good = [nn.Parameter(t.randn(3)), nn.Parameter(t.randn(2, 2))]\n"
            "out = ex2_audit_optim_params(good)\n"
            "assert out == [], f'all-safe case should return [], got {out}'\n"
            "\n"
            "# Three params: one safe, one non-leaf, one no-grad.\n"
            "p_safe = nn.Parameter(t.randn(4))                       # safe (skip)\n"
            "p_non_leaf = nn.Parameter(t.randn(3)) * 2               # non-leaf, grad-on\n"
            "p_no_grad = t.randn(5)                                  # leaf, grad-off\n"
            "# Sanity-check the test fixtures so the premise is right.\n"
            "assert p_non_leaf.is_leaf is False and p_non_leaf.requires_grad is True\n"
            "assert p_no_grad.is_leaf is True and p_no_grad.requires_grad is False\n"
            "\n"
            "report = ex2_audit_optim_params([p_safe, p_non_leaf, p_no_grad])\n"
            "assert isinstance(report, list), 'must return list'\n"
            "assert len(report) == 2, (\n"
            "    f'safe param should be skipped, got {len(report)} problems instead of 2'\n"
            ")\n"
            "\n"
            "# Verify each problem dict.\n"
            "by_index = {r['index']: r for r in report}\n"
            "assert set(by_index.keys()) == {1, 2}, (\n"
            "    f'problem indices must be {{1, 2}} (skipping safe index 0), got {sorted(by_index.keys())}'\n"
            ")\n"
            "\n"
            "# Index 1: non-leaf.\n"
            "r1 = by_index[1]\n"
            "assert r1['category'] == 'non-leaf', f'index 1 category: {r1!r}'\n"
            "assert r1['is_leaf'] is False\n"
            "assert r1['requires_grad'] is True\n"
            "assert r1['shape'] == (3,)\n"
            "\n"
            "# Index 2: no-grad.\n"
            "r2 = by_index[2]\n"
            "assert r2['category'] == 'no-grad', f'index 2 category: {r2!r}'\n"
            "assert r2['is_leaf'] is True\n"
            "assert r2['requires_grad'] is False\n"
            "assert r2['shape'] == (5,)\n"
            "\n"
            "# Order preservation.\n"
            "assert [r['index'] for r in report] == sorted([r['index'] for r in report]), (\n"
            "    'problems should appear in source-list order'\n"
            ")\n"
            "\n"
            "# Empty input.\n"
            "assert ex2_audit_optim_params([]) == [], 'empty input → empty report'\n"
            "\n"
            "# Mixed bigger list — interleaved safe/bad — verify non-bad indices skipped cleanly.\n"
            "params = [\n"
            "    nn.Parameter(t.randn(2)),          # 0 safe\n"
            "    t.randn(2),                         # 1 no-grad\n"
            "    nn.Parameter(t.randn(2)),          # 2 safe\n"
            "    nn.Parameter(t.randn(2)) * 2,      # 3 non-leaf\n"
            "    nn.Parameter(t.randn(2)),          # 4 safe\n"
            "]\n"
            "rep = ex2_audit_optim_params(params)\n"
            "assert len(rep) == 2, f'expected 2 problems out of 5, got {len(rep)}'\n"
            "indices = {r['index']: r['category'] for r in rep}\n"
            "assert indices == {1: 'no-grad', 3: 'non-leaf'}, f'mismatch: {indices}'"
        ),
        "solution_body": (
            "def ex2_audit_optim_params(params):\n"
            "    problems = []\n"
            "    for i, p in enumerate(params):\n"
            "        bad_leaf = not p.is_leaf\n"
            "        bad_grad = not p.requires_grad\n"
            "        if not bad_leaf and not bad_grad:\n"
            "            continue\n"
            "        # Non-leaf is the more fundamental break — report it first.\n"
            "        category = 'non-leaf' if bad_leaf else 'no-grad'\n"
            "        problems.append({\n"
            "            'index': i,\n"
            "            'shape': tuple(p.shape),\n"
            "            'is_leaf': bool(p.is_leaf),\n"
            "            'requires_grad': bool(p.requires_grad),\n"
            "            'category': category,\n"
            "        })\n"
            "    return problems"
        ),
        "solution_notes": (
            "**Why a report instead of an assertion.** The ex1 assertion "
            "is correct for a TIGHT defensive check at optimizer "
            "construction — fail fast on the first problem. This "
            "deepening drill is for the DEBUGGING workflow: you've already "
            "hit a silent-no-op bug and want to see every offender in one "
            "pass. Categorizing them tells you which fix to apply: "
            "non-leaf usually means 'you `.to(device)`'d a Parameter — "
            "move the Module instead'; no-grad usually means 'you froze "
            "this Parameter intentionally — exclude it from the optimizer "
            "list'.\n\n"
            "**Why only two categories, not three.** PyTorch enforces "
            "that any tensor with `requires_grad=False` is automatically "
            "a leaf (by docs: 'tensors that have requires_grad=False will "
            "be leaf Tensors by convention'). So a 'non-leaf + no-grad' "
            "specimen can't be constructed via normal means — operating "
            "on a Parameter inside `no_grad` yields a LEAF tensor with "
            "`requires_grad=False`. Reporting non-leaf takes priority "
            "because it's the more fundamental break — even if you fixed "
            "`requires_grad`, the optimizer still couldn't update a "
            "non-leaf.\n\n"
            "**Why non-leaf priority.** When both flags are bad (rare in "
            "practice, but possible with custom autograd ops), saying "
            "'non-leaf' first directs the user to the structural fix "
            "('move the whole Module, not the parameter'), which usually "
            "restores `requires_grad=True` as a side effect."
        ),
        "extra_imports": [],
    },

    # ===================================================================
    # t-stack-trajectory  —  ex2
    # ex1 = stack list to (B, T, D). NEW facet: given a (B, T, D)
    # trajectory, extract a time window then return per-step deltas.
    # Tests slicing along dim=1 and `traj[:, 1:] - traj[:, :-1]`.
    # Bloom: Apply.
    # ===================================================================
    {
        "atom_id": "t-stack-trajectory",
        "subtopic": "Generative: torch.stack trajectory",
        "topic_folder": TOPIC_GENERATIVE,
        "atom_recap_md": RECAP_STACK_TRAJ,
        "exercise_index": 2,
        "exercise_title": "window-slice a (B,T,D) trajectory and return per-step deltas",
        "slug": "window-slice-a-trajectory-and-return-per-step-deltas",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["trajectory", "slicing", "delta", "time-axis"],
        "kcs": [
            "slice-along-time-axis-dim1",
            "consecutive-step-delta",
        ],
        "lo": (
            "Apply trajectory slicing (`traj[:, t0:t1, :]`) plus consecutive-"
            "difference (`win[:, 1:] - win[:, :-1]`) to extract a windowed "
            "per-step delta tensor of shape `(B, t1 - t0 - 1, D)`."
        ),
        "prompt_body": (
            "Implement `ex2_window_deltas(traj, t0, t1)`. Given a "
            "`(B, T, D)` trajectory tensor, extract the time window "
            "`[t0, t1)` (Python half-open convention) and return the "
            "per-step deltas WITHIN that window:\n\n"
            "1. Slice: `window = traj[:, t0:t1, :]` — shape `(B, t1 - t0, D)`.\n"
            "2. Compute deltas along the time axis: "
            "`deltas = window[:, 1:, :] - window[:, :-1, :]` — shape "
            "`(B, t1 - t0 - 1, D)`.\n"
            "3. Return `deltas`.\n\n"
            "**Edge cases.**\n"
            "- `t1 - t0 == 1`: the window has a single timestep ⇒ deltas "
            "has shape `(B, 0, D)`. That's the natural empty case — return "
            "it as-is, do NOT raise.\n"
            "- `t1 - t0 == 0`: empty window ⇒ deltas shape `(B, 0, D)`. "
            "(Slicing handles this — the subtraction is over empty slices.)\n"
            "- `t0 == 0, t1 == T`: full-trajectory deltas, shape "
            "`(B, T - 1, D)`. This is the common case.\n\n"
            "**Do NOT use a Python loop.** Slice + subtract is one tensor "
            "operation each.\n\n"
            "Inputs:\n"
            "- `traj`: `(B, T, D)` float tensor.\n"
            "- `t0`, `t1`: ints with `0 <= t0 <= t1 <= T`.\n\n"
            "Output: `(B, t1 - t0 - 1, D)` float tensor (with `max(0, ...)`)."
        ),
        "stub": (
            "def ex2_window_deltas(traj: Tensor, t0: int, t1: int) -> Tensor:\n"
            '    """Slice [t0:t1) along time, then return per-step deltas."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Tiny exact case.\n"
            "traj = t.tensor([\n"
            "    [[1.0, 2.0], [3.0, 4.0], [6.0, 8.0], [10.0, 14.0]],\n"
            "    [[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [6.0,  6.0]],\n"
            "])  # (B=2, T=4, D=2)\n"
            "\n"
            "# Full window — all consecutive deltas.\n"
            "d_full = ex2_window_deltas(traj, 0, 4)\n"
            "assert d_full.shape == (2, 3, 2), f'expected (2,3,2), got {tuple(d_full.shape)}'\n"
            "expected_full = t.tensor([\n"
            "    [[2.0, 2.0], [3.0, 4.0], [4.0, 6.0]],\n"
            "    [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],\n"
            "])\n"
            "assert t.allclose(d_full, expected_full), f'full-window deltas wrong:\\n{d_full}'\n"
            "\n"
            "# Sub-window [1, 3) → 2 timesteps → 1 delta.\n"
            "d_sub = ex2_window_deltas(traj, 1, 3)\n"
            "assert d_sub.shape == (2, 1, 2), f'expected (2,1,2), got {tuple(d_sub.shape)}'\n"
            "expected_sub = t.tensor([\n"
            "    [[3.0, 4.0]],   # 6-3, 8-4\n"
            "    [[2.0, 2.0]],   # 3-1, 3-1\n"
            "])\n"
            "assert t.allclose(d_sub, expected_sub), f'sub-window deltas wrong:\\n{d_sub}'\n"
            "\n"
            "# Single-timestep window → empty deltas.\n"
            "d_one = ex2_window_deltas(traj, 2, 3)\n"
            "assert d_one.shape == (2, 0, 2), f'single-step window must give 0 deltas, got {tuple(d_one.shape)}'\n"
            "\n"
            "# Empty window → also empty deltas.\n"
            "d_empty = ex2_window_deltas(traj, 2, 2)\n"
            "assert d_empty.shape == (2, 0, 2), f'empty window must give shape (B,0,D), got {tuple(d_empty.shape)}'\n"
            "\n"
            "# Realistic shape.\n"
            "rng = t.Generator().manual_seed(0)\n"
            "B, T, D = 8, 30, 6\n"
            "big = t.randn(B, T, D, generator=rng)\n"
            "d = ex2_window_deltas(big, 5, 20)\n"
            "assert d.shape == (B, 14, D), f'expected ({B},14,{D}), got {tuple(d.shape)}'\n"
            "# Spot check.\n"
            "expected_first = big[:, 6, :] - big[:, 5, :]\n"
            "assert t.allclose(d[:, 0, :], expected_first, atol=1e-6), 'first delta mismatch'\n"
            "expected_last = big[:, 19, :] - big[:, 18, :]\n"
            "assert t.allclose(d[:, -1, :], expected_last, atol=1e-6), 'last delta mismatch'\n"
            "\n"
            "# Type preservation.\n"
            "assert d.dtype == big.dtype, f'dtype must propagate, got {d.dtype} vs {big.dtype}'"
        ),
        "solution_body": (
            "def ex2_window_deltas(traj, t0, t1):\n"
            "    window = traj[:, t0:t1, :]\n"
            "    return window[:, 1:, :] - window[:, :-1, :]"
        ),
        "solution_notes": (
            "**Why slicing handles all the edge cases for free.** "
            "`traj[:, 2:2, :]` gives a `(B, 0, D)` tensor — empty along "
            "`dim=1`. `window[:, 1:, :]` and `window[:, :-1, :]` on an "
            "empty window are also empty; their difference is `(B, 0, D)`. "
            "Single-step `[2:3]` gives `(B, 1, D)`; `[1:]` is `(B, 0, D)` "
            "and `[:-1]` is `(B, 0, D)`; difference is `(B, 0, D)`. No "
            "branches needed.\n\n"
            "**Why `dim=1` slicing, not `dim=0`.** The `(B, T, D)` "
            "convention puts time at `dim=1`. Slicing `traj[t0:t1]` "
            "(without the `:`) would slice the BATCH axis — silently "
            "wrong and a common bug. Always write the full `traj[:, t0:t1, :]` "
            "or `traj[:, t0:t1]` for clarity.\n\n"
            "**Generalization.** The same `x[..., 1:] - x[..., :-1]` "
            "pattern computes consecutive deltas along ANY trailing axis. "
            "For `(B, T, D)`, you want `dim=1`; for a sequence at `dim=0` "
            "use `x[1:] - x[:-1]`. PyTorch also offers `t.diff(x, dim=1)` "
            "which is a one-line wrapper for the same op."
        ),
        "extra_imports": [],
    },

    # ===================================================================
    # two-optimizers-alternating-step  —  ex2
    # ex1 = one D-step + one G-step. NEW facet: WGAN-style n_critic
    # D-steps before each G-step. Tests that param-count moved on D is
    # n_critic times what moved on G. Bloom: Apply.
    # ===================================================================
    {
        "atom_id": "two-optimizers-alternating-step",
        "subtopic": "GAN: Two-optimizers alternating step",
        "topic_folder": TOPIC_GENERATIVE,
        "atom_recap_md": RECAP_TWO_OPTIMIZERS,
        "exercise_index": 2,
        "exercise_title": "WGAN-style n_critic D-steps per G-step",
        "slug": "wgan-style-n-critic-d-steps-per-g-step",
        "bloom_level": "Apply",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["gan", "wgan", "n-critic", "d-step-loop"],
        "kcs": [
            "k-d-steps-per-g-step",
            "per-step-fresh-z-sampling",
        ],
        "lo": (
            "Apply the WGAN-style training pattern of `n_critic` D-step "
            "updates followed by 1 G-step update per outer iteration, with "
            "fresh noise per inner step."
        ),
        "prompt_body": (
            "Implement `ex2_wgan_iter(G, D, G_opt, D_opt, x_real, z_dim, n_critic)`.\n\n"
            "One outer iteration of the WGAN training loop:\n\n"
            "**Inner loop — `n_critic` D-step updates:**\n"
            "  For each of the `n_critic` inner steps:\n"
            "    a. `D_opt.zero_grad()`\n"
            "    b. Sample FRESH `z = t.randn(B, z_dim)` (NOT reuse across steps).\n"
            "    c. `fake = G(z).detach()` — stop-grad into G.\n"
            "    d. `loss_D = (D(fake) - D(x_real)).mean()` (simplified Wasserstein-style).\n"
            "    e. `loss_D.backward()`\n"
            "    f. `D_opt.step()`\n"
            "    g. Record `loss_D.item()` in a list.\n\n"
            "**Outer — 1 G-step update:**\n"
            "  a. `G_opt.zero_grad()`\n"
            "  b. Sample FRESH `z` (independent of the inner-loop noises).\n"
            "  c. `fake = G(z)` — NO detach, grad flows into G.\n"
            "  d. `loss_G = -D(fake).mean()`\n"
            "  e. `loss_G.backward()`\n"
            "  f. `G_opt.step()`\n\n"
            "Return `(d_losses, loss_G_value)` — a Python list of `n_critic` "
            "floats and a single float.\n\n"
            "Infer `B = x_real.shape[0]` from the input.\n\n"
            "Inputs:\n"
            "- `G`, `D`: `nn.Module`s.\n"
            "- `G_opt`, `D_opt`: optimizers wired to G.parameters() / D.parameters() respectively.\n"
            "- `x_real`: real samples, shape `(B, x_dim)`.\n"
            "- `z_dim`: noise dimension (int).\n"
            "- `n_critic`: number of D-step updates per G-step (int, >=1).\n\n"
            "Output: `(list[float], float)`."
        ),
        "stub": (
            "def ex2_wgan_iter(G, D, G_opt, D_opt, x_real, z_dim, n_critic):\n"
            '    """One outer WGAN iteration: n_critic D-steps + 1 G-step. Return (d_losses, loss_G)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import torch.nn as nn\n"
            "\n"
            "t.manual_seed(0)\n"
            "z_dim, x_dim, B = 4, 6, 8\n"
            "G = nn.Sequential(nn.Linear(z_dim, 16), nn.ReLU(), nn.Linear(16, x_dim))\n"
            "D = nn.Sequential(nn.Linear(x_dim, 16), nn.ReLU(), nn.Linear(16, 1))\n"
            "G_opt = t.optim.SGD(G.parameters(), lr=1e-2)\n"
            "D_opt = t.optim.SGD(D.parameters(), lr=1e-2)\n"
            "x_real = t.randn(B, x_dim)\n"
            "\n"
            "n_critic = 5\n"
            "G_before = {n: p.detach().clone() for n, p in G.named_parameters()}\n"
            "D_before = {n: p.detach().clone() for n, p in D.named_parameters()}\n"
            "\n"
            "d_losses, loss_G_val = ex2_wgan_iter(G, D, G_opt, D_opt, x_real, z_dim, n_critic)\n"
            "\n"
            "# Return shape.\n"
            "assert isinstance(d_losses, list), f'd_losses must be list, got {type(d_losses).__name__}'\n"
            "assert len(d_losses) == n_critic, (\n"
            "    f'expected {n_critic} D-step losses, got {len(d_losses)}'\n"
            ")\n"
            "for dl in d_losses:\n"
            "    assert isinstance(dl, float), f'each D-loss must be float, got {type(dl).__name__}'\n"
            "    assert t.isfinite(t.tensor(dl)).item(), f'D-loss non-finite: {dl}'\n"
            "assert isinstance(loss_G_val, float), f'loss_G must be float, got {type(loss_G_val).__name__}'\n"
            "assert t.isfinite(t.tensor(loss_G_val)).item(), f'G-loss non-finite: {loss_G_val}'\n"
            "\n"
            "# Both modules moved.\n"
            "G_moved = any(not t.allclose(G_before[n], p.detach()) for n, p in G.named_parameters())\n"
            "D_moved = any(not t.allclose(D_before[n], p.detach()) for n, p in D.named_parameters())\n"
            "assert G_moved, 'G params did not move — did you call G_opt.step()?'\n"
            "assert D_moved, 'D params did not move — did you run the D-step inner loop?'\n"
            "\n"
            "# n_critic=1 must collapse to the vanilla pattern (one D-step + one G-step).\n"
            "t.manual_seed(1)\n"
            "G2 = nn.Sequential(nn.Linear(z_dim, 16), nn.ReLU(), nn.Linear(16, x_dim))\n"
            "D2 = nn.Sequential(nn.Linear(x_dim, 16), nn.ReLU(), nn.Linear(16, 1))\n"
            "G2_opt = t.optim.SGD(G2.parameters(), lr=1e-2)\n"
            "D2_opt = t.optim.SGD(D2.parameters(), lr=1e-2)\n"
            "x_real2 = t.randn(B, x_dim)\n"
            "d_losses_1, _ = ex2_wgan_iter(G2, D2, G2_opt, D2_opt, x_real2, z_dim, n_critic=1)\n"
            "assert len(d_losses_1) == 1, f'n_critic=1 should give exactly 1 D-loss, got {len(d_losses_1)}'\n"
            "\n"
            "# Stability across 20 outer iterations with n_critic=3.\n"
            "t.manual_seed(2)\n"
            "G3 = nn.Sequential(nn.Linear(z_dim, 16), nn.ReLU(), nn.Linear(16, x_dim))\n"
            "D3 = nn.Sequential(nn.Linear(x_dim, 16), nn.ReLU(), nn.Linear(16, 1))\n"
            "G3_opt = t.optim.SGD(G3.parameters(), lr=1e-3)\n"
            "D3_opt = t.optim.SGD(D3.parameters(), lr=1e-3)\n"
            "for _ in range(20):\n"
            "    x_r = t.randn(B, x_dim)\n"
            "    dls, lg = ex2_wgan_iter(G3, D3, G3_opt, D3_opt, x_r, z_dim, n_critic=3)\n"
            "    assert len(dls) == 3\n"
            "    for dl in dls:\n"
            "        assert t.isfinite(t.tensor(dl)).item()\n"
            "    assert t.isfinite(t.tensor(lg)).item()\n"
            "for p in G3.parameters():\n"
            "    assert t.isfinite(p).all(), 'G params went non-finite over 20 iters'\n"
            "for p in D3.parameters():\n"
            "    assert t.isfinite(p).all(), 'D params went non-finite over 20 iters'\n"
            "\n"
            "# Param isolation between optimizers (no cross-wiring).\n"
            "G3_ids = {id(p) for g in G3_opt.param_groups for p in g['params']}\n"
            "for p in D3.parameters():\n"
            "    assert id(p) not in G3_ids, 'CROSS-WIRING: D param in G_opt'"
        ),
        "solution_body": (
            "def ex2_wgan_iter(G, D, G_opt, D_opt, x_real, z_dim, n_critic):\n"
            "    B = x_real.shape[0]\n"
            "    d_losses = []\n"
            "    # === inner loop: n_critic D-steps ===\n"
            "    for _ in range(n_critic):\n"
            "        D_opt.zero_grad()\n"
            "        z = t.randn(B, z_dim)\n"
            "        fake = G(z).detach()\n"
            "        loss_D = (D(fake) - D(x_real)).mean()\n"
            "        loss_D.backward()\n"
            "        D_opt.step()\n"
            "        d_losses.append(loss_D.item())\n"
            "    # === outer: 1 G-step ===\n"
            "    G_opt.zero_grad()\n"
            "    z = t.randn(B, z_dim)\n"
            "    fake = G(z)\n"
            "    loss_G = -D(fake).mean()\n"
            "    loss_G.backward()\n"
            "    G_opt.step()\n"
            "    return d_losses, loss_G.item()"
        ),
        "solution_notes": (
            "**Why fresh `z` per inner step.** Each D-step asks D to "
            "distinguish a NEW fake from the real data. Reusing the same "
            "`z` would let D memorize one particular noise sample's "
            "shortcomings — defeats the point of the critic loop, which "
            "is to give D a strong signal over the FAKE DISTRIBUTION, not "
            "one fake point.\n\n"
            "**Why `n_critic` D-steps for WGAN.** The Wasserstein critic "
            "needs to be close to optimal at each iteration for the EM "
            "distance estimate to be reliable. The original WGAN paper "
            "uses 5; WGAN-GP also uses 5. With `n_critic=1` you recover "
            "the Goodfellow vanilla alternation — useful for sanity "
            "checks (the code path is the same, just with one inner "
            "iteration).\n\n"
            "**Per-optimizer zero_grad.** `D_opt.zero_grad()` inside the "
            "inner loop, `G_opt.zero_grad()` once outside. Calling each "
            "optimizer's `zero_grad()` right before its `step()` is the "
            "convention that keeps the code readable as you scale up "
            "n_critic — never wonder which set of grads is being cleared."
        ),
        "extra_imports": [],
    },

    # ===================================================================
    # cross-product-normal  —  ex2
    # ex1 = single triangle. NEW facet: batched (N, 3, 3) triangles +
    # degenerate-triangle bool mask. Bloom: Apply.
    # ===================================================================
    {
        "atom_id": "cross-product-normal",
        "subtopic": "Geometry: Cross-product surface normal",
        "topic_folder": TOPIC_GEOMETRY_CNN,
        "atom_recap_md": RECAP_CROSS_NORMAL,
        "exercise_index": 2,
        "exercise_title": "batched unit normals + degenerate-triangle mask",
        "slug": "batched-unit-normals-and-degenerate-mask",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["cross-product", "batched", "degenerate", "mask"],
        "kcs": [
            "batched-cross-along-dim-minus-1",
            "degenerate-norm-mask",
        ],
        "lo": (
            "Apply `t.linalg.cross(..., dim=-1)` over a `(N, 3, 3)` batch "
            "of triangle vertices and produce both the unit-normals tensor "
            "AND a boolean mask of degenerate (colinear) triangles."
        ),
        "prompt_body": (
            "Implement `ex2_batched_normals(tris, eps=1e-8)`. Compute unit "
            "surface normals for a batch of triangles AND flag the "
            "degenerate ones.\n\n"
            "1. `tris` has shape `(N, 3, 3)` — N triangles, 3 vertices each, "
            "3-D coords.\n"
            "2. Edges from shared vertex `P1 = tris[:, 0]`:\n"
            "   ```python\n"
            "   e1 = tris[:, 1] - tris[:, 0]   # (N, 3)\n"
            "   e2 = tris[:, 2] - tris[:, 0]   # (N, 3)\n"
            "   ```\n"
            "3. Cross product along the LAST axis: "
            "`n = t.linalg.cross(e1, e2, dim=-1)`. Shape `(N, 3)`.\n"
            "4. Norms: `norms = n.norm(dim=-1, keepdim=True)` — shape "
            "`(N, 1)`.\n"
            "5. Degenerate mask: `valid = (norms.squeeze(-1) > eps)` — "
            "shape `(N,)`, `True` iff the triangle has nonzero area.\n"
            "6. Normalize SAFELY. For degenerate triangles, division would "
            "produce `nan` / `inf`; replace the denominator with `1.0` where "
            "the triangle is invalid (the resulting normal there is the "
            "zero vector — a sentinel that downstream code can detect).\n"
            "   ```python\n"
            "   safe_norms = norms.clamp(min=eps)\n"
            "   unit = n / safe_norms\n"
            "   # zero-out the degenerate ones so the value is a clean sentinel\n"
            "   unit[~valid] = 0.0\n"
            "   ```\n"
            "7. Return `(unit, valid)`. `unit` has shape `(N, 3)`; `valid` "
            "has shape `(N,)` (bool).\n\n"
            "**Do NOT** call `ex1_triangle_normal` in a Python loop. Use "
            "batched ops throughout.\n\n"
            "Input: `tris` shape `(N, 3, 3)` float; `eps` float.\n"
            "Output: tuple `(unit_normals (N,3), valid_mask (N,))`."
        ),
        "stub": (
            "def ex2_batched_normals(tris: Tensor, eps: float = 1e-8) -> tuple:\n"
            '    """Return (unit_normals (N,3), valid_mask (N,)) for a batch of triangles."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import math\n"
            "\n"
            "# Mixed batch: 3 valid + 1 degenerate (colinear).\n"
            "tris = t.tensor([\n"
            "    # CCW in z=0 plane → +z normal\n"
            "    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],\n"
            "    # x=5 plane → +x normal\n"
            "    [[5.0, 0.0, 0.0], [5.0, 1.0, 0.0], [5.0, 0.0, 1.0]],\n"
            "    # tilted triangle\n"
            "    [[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 3.0]],\n"
            "    # DEGENERATE: three colinear points along x-axis\n"
            "    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],\n"
            "])\n"
            "unit, valid = ex2_batched_normals(tris)\n"
            "assert unit.shape == (4, 3), f'expected (4, 3), got {tuple(unit.shape)}'\n"
            "assert valid.shape == (4,), f'expected (4,), got {tuple(valid.shape)}'\n"
            "assert valid.dtype == t.bool, f'mask must be bool, got {valid.dtype}'\n"
            "\n"
            "# Valid mask: first 3 True, last False.\n"
            "expected_valid = t.tensor([True, True, True, False])\n"
            "assert t.equal(valid, expected_valid), f'valid mask wrong: {valid}'\n"
            "\n"
            "# Triangle 0: +z.\n"
            "assert t.allclose(unit[0], t.tensor([0.0, 0.0, 1.0]), atol=1e-5), f'tri 0 normal: {unit[0]}'\n"
            "# Triangle 1: +x.\n"
            "assert t.allclose(unit[1], t.tensor([1.0, 0.0, 0.0]), atol=1e-5), f'tri 1 normal: {unit[1]}'\n"
            "# Triangle 2: must be unit length and perpendicular to both edges.\n"
            "n2 = unit[2]\n"
            "assert abs(n2.norm().item() - 1.0) < 1e-5, f'tri 2 not unit: |n|={n2.norm().item()}'\n"
            "e1_2 = tris[2, 1] - tris[2, 0]\n"
            "e2_2 = tris[2, 2] - tris[2, 0]\n"
            "assert abs((n2 * e1_2).sum().item()) < 1e-5, f'tri 2 not perp to e1: dot={(n2*e1_2).sum().item()}'\n"
            "assert abs((n2 * e2_2).sum().item()) < 1e-5, f'tri 2 not perp to e2: dot={(n2*e2_2).sum().item()}'\n"
            "# Triangle 3: degenerate → zero-vector sentinel.\n"
            "assert t.allclose(unit[3], t.zeros(3), atol=1e-7), (\n"
            "    f'degenerate triangle must have zero-vector normal (sentinel), got {unit[3]}'\n"
            ")\n"
            "# No nan / inf anywhere — that's the whole point of the safe-clamp pattern.\n"
            "assert t.isfinite(unit).all(), 'normals must be finite; check the safe-divide pattern'\n"
            "\n"
            "# Stress test: 50 random valid triangles + 5 degenerate.\n"
            "rng = t.Generator().manual_seed(0)\n"
            "rand_valid = t.randn(50, 3, 3, generator=rng)\n"
            "# Construct 5 colinear ones explicitly.\n"
            "ts_col = []\n"
            "for k in range(5):\n"
            "    a = t.randn(3, generator=rng)\n"
            "    b = t.randn(3, generator=rng)\n"
            "    ts_col.append(t.stack([a, a + b, a + 2 * b]))   # all on line through a\n"
            "rand_col = t.stack(ts_col)\n"
            "big = t.cat([rand_valid, rand_col], dim=0)\n"
            "# Use a slightly larger eps to catch the residual float32 error in the\n"
            "# colinear construction (cross of nearly-parallel edges leaves ~1e-7 noise).\n"
            "unit_b, valid_b = ex2_batched_normals(big, eps=1e-5)\n"
            "assert unit_b.shape == (55, 3)\n"
            "assert valid_b.shape == (55,)\n"
            "assert valid_b[:50].all(), 'random triangles should be valid (probability 1)'\n"
            "assert (~valid_b[50:]).all(), 'all 5 colinear triangles must be flagged degenerate'\n"
            "# Unit-length for the valid ones.\n"
            "valid_norms = unit_b[valid_b].norm(dim=-1)\n"
            "assert t.allclose(valid_norms, t.ones_like(valid_norms), atol=1e-5), (\n"
            "    f'valid normals must be unit length, got norms ranging {valid_norms.min().item():.4f}..{valid_norms.max().item():.4f}'\n"
            ")\n"
            "# Zero for the degenerate ones.\n"
            "assert t.allclose(unit_b[~valid_b], t.zeros_like(unit_b[~valid_b]), atol=1e-7)"
        ),
        "solution_body": (
            "def ex2_batched_normals(tris, eps=1e-8):\n"
            "    e1 = tris[:, 1] - tris[:, 0]                   # (N, 3)\n"
            "    e2 = tris[:, 2] - tris[:, 0]                   # (N, 3)\n"
            "    n = t.linalg.cross(e1, e2, dim=-1)             # (N, 3)\n"
            "    norms = n.norm(dim=-1, keepdim=True)           # (N, 1)\n"
            "    valid = norms.squeeze(-1) > eps                # (N,) bool\n"
            "    safe = norms.clamp(min=eps)\n"
            "    unit = n / safe\n"
            "    unit[~valid] = 0.0                             # sentinel\n"
            "    return unit, valid"
        ),
        "solution_notes": (
            "**Why `dim=-1` instead of `dim=1`.** Either works for a "
            "`(N, 3)` tensor. `dim=-1` is the convention that survives "
            "rank changes — the same code works if you later wrap "
            "everything in another batch dim (`(M, N, 3, 3)`).\n\n"
            "**Why `clamp(min=eps)` then mask, not just `if-else`.** "
            "Branching per-element would require a Python loop or a "
            "`torch.where` chain. Clamping the denominator first ensures "
            "the division never produces `nan`/`inf`; then a single bool "
            "indexer (`unit[~valid] = 0.0`) replaces the bogus values "
            "with a clean sentinel. Vectorized + numerically safe.\n\n"
            "**Why a zero vector as sentinel.** Downstream code can "
            "detect 'this normal is invalid' with `(unit == 0).all(-1)`, "
            "which is cheap to vectorize. A unit-length sentinel (e.g. "
            "+z) would confuse downstream lighting that takes `dot(L, n)` "
            "— a real, non-degenerate +z triangle would look identical "
            "to a degenerate one. The zero vector has no valid "
            "interpretation, so it's safely distinguishable."
        ),
        "extra_imports": [],
    },

    # ===================================================================
    # rotation-matrix-3d  —  ex2
    # ex1 = build Rodrigues R. NEW facet: compose two rotations and
    # verify the composition is itself a rotation (orthogonal + det=1).
    # Also check non-commutativity. Bloom: Analyze.
    # ===================================================================
    {
        "atom_id": "rotation-matrix-3d",
        "subtopic": "Geometry: Rotation matrix 3-D (full)",
        "topic_folder": TOPIC_GEOMETRY_CNN,
        "atom_recap_md": RECAP_ROT3D,
        "exercise_index": 2,
        "exercise_title": "compose two Rodrigues rotations and verify SO(3) closure",
        "slug": "compose-two-rodrigues-rotations-and-verify-so3-closure",
        "bloom_level": "Analyze",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["rotation", "composition", "SO(3)", "non-commutative"],
        "kcs": [
            "rotation-composition-matmul",
            "verify-orthogonal-and-det-one",
        ],
        "lo": (
            "Analyze the composition of two Rodrigues rotations by computing "
            "`R_total = R2 @ R1`, checking orthogonality + determinant, and "
            "exposing non-commutativity by comparing against `R1 @ R2`."
        ),
        "prompt_body": (
            "A helper `rot3d(axis, theta)` is provided in the stub — it "
            "builds the Rodrigues 3-D rotation matrix exactly as in ex1.\n\n"
            "Implement `ex2_compose_and_check(axis1, theta1, axis2, theta2)` "
            "which returns a dict with these keys:\n\n"
            "- `'R1'`: the (3, 3) rotation from `(axis1, theta1)`.\n"
            "- `'R2'`: the (3, 3) rotation from `(axis2, theta2)`.\n"
            "- `'R_total'`: the composition `R2 @ R1` (apply R1 FIRST, then R2). "
            "Shape `(3, 3)`.\n"
            "- `'R_swapped'`: the OTHER composition order, `R1 @ R2`. Shape `(3, 3)`.\n"
            "- `'is_orthogonal'`: `True` iff `R_total @ R_total.T ≈ I` "
            "(`atol=1e-5`). Python bool.\n"
            "- `'det_close_to_one'`: `True` iff `|det(R_total) - 1| < 1e-5`. Python bool.\n"
            "- `'commutes'`: `True` iff `R_total ≈ R_swapped` element-wise "
            "(`atol=1e-5`). Python bool.\n\n"
            "All four matrix values stay as `(3, 3)` `float32` tensors.\n\n"
            "Inputs:\n"
            "- `axis1`, `axis2`: `(3,)` float tensors (NOT assumed unit).\n"
            "- `theta1`, `theta2`: scalar float angles in radians.\n\n"
            "Output: dict as described above."
        ),
        "stub": (
            "import math\n"
            "\n"
            "def rot3d(axis: Tensor, theta: float) -> Tensor:\n"
            '    """Rodrigues 3-D rotation matrix for axis (any length) by theta radians."""\n'
            "    k = axis / axis.norm()\n"
            "    kx, ky, kz = k[0], k[1], k[2]\n"
            "    K = t.stack([\n"
            "        t.stack([t.zeros_like(kx),          -kz,                  ky]),\n"
            "        t.stack([                 kz, t.zeros_like(kx),          -kx]),\n"
            "        t.stack([                -ky,                kx, t.zeros_like(kx)]),\n"
            "    ])\n"
            "    s, c = math.sin(theta), math.cos(theta)\n"
            "    return t.eye(3) + s * K + (1 - c) * (K @ K)\n"
            "\n"
            "\n"
            "def ex2_compose_and_check(axis1: Tensor, theta1: float,\n"
            "                          axis2: Tensor, theta2: float) -> dict:\n"
            '    """Return dict with R1, R2, R_total, R_swapped, is_orthogonal, det_close_to_one, commutes."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import math\n"
            "\n"
            "# Case 1: two different axes — non-commuting.\n"
            "axis1 = t.tensor([0.0, 0.0, 1.0])   # +z\n"
            "theta1 = math.pi / 2                # 90°\n"
            "axis2 = t.tensor([1.0, 0.0, 0.0])   # +x\n"
            "theta2 = math.pi / 3                # 60°\n"
            "out = ex2_compose_and_check(axis1, theta1, axis2, theta2)\n"
            "assert isinstance(out, dict), 'must return dict'\n"
            "for key in ['R1', 'R2', 'R_total', 'R_swapped', 'is_orthogonal', 'det_close_to_one', 'commutes']:\n"
            "    assert key in out, f'missing key: {key}'\n"
            "\n"
            "# Shape checks.\n"
            "for key in ['R1', 'R2', 'R_total', 'R_swapped']:\n"
            "    assert out[key].shape == (3, 3), f'{key} shape: {tuple(out[key].shape)}'\n"
            "\n"
            "# R_total = R2 @ R1.\n"
            "expected_total = out['R2'] @ out['R1']\n"
            "assert t.allclose(out['R_total'], expected_total, atol=1e-5), (\n"
            "    f'R_total must equal R2 @ R1 (apply R1 first, then R2)'\n"
            ")\n"
            "# R_swapped = R1 @ R2.\n"
            "expected_swapped = out['R1'] @ out['R2']\n"
            "assert t.allclose(out['R_swapped'], expected_swapped, atol=1e-5), (\n"
            "    f'R_swapped must equal R1 @ R2'\n"
            ")\n"
            "\n"
            "# SO(3) closure: R_total is itself orthogonal with det +1.\n"
            "assert out['is_orthogonal'] is True, 'R2 @ R1 must be orthogonal'\n"
            "assert out['det_close_to_one'] is True, 'det(R2 @ R1) must be +1'\n"
            "\n"
            "# Non-commutativity: rotations around DIFFERENT axes don't commute.\n"
            "assert out['commutes'] is False, (\n"
            "    'z-rotation and x-rotation should NOT commute — '\n"
            "    'check that you computed both R_total and R_swapped correctly'\n"
            ")\n"
            "\n"
            "# Case 2: same-axis rotations DO commute (R(theta_a) @ R(theta_b) = R(theta_a + theta_b)).\n"
            "axis_z = t.tensor([0.0, 0.0, 1.0])\n"
            "out2 = ex2_compose_and_check(axis_z, 0.4, axis_z, 0.7)\n"
            "assert out2['commutes'] is True, 'same-axis rotations must commute'\n"
            "assert out2['is_orthogonal'] is True\n"
            "assert out2['det_close_to_one'] is True\n"
            "# Same-axis composition equals a single rotation by the sum of angles.\n"
            "R_sum = t.eye(3)\n"
            "# Recompute via the helper exposed in the stub namespace.\n"
            "R_combined = rot3d(axis_z, 0.4 + 0.7)\n"
            "assert t.allclose(out2['R_total'], R_combined, atol=1e-5), (\n"
            "    'same-axis composition should equal a single rotation by the angle sum'\n"
            ")\n"
            "\n"
            "# Case 3: identity composition (theta=0 for both) → R_total ≈ I.\n"
            "out3 = ex2_compose_and_check(t.tensor([1.0, 0.0, 0.0]), 0.0,\n"
            "                             t.tensor([0.0, 1.0, 0.0]), 0.0)\n"
            "assert t.allclose(out3['R_total'], t.eye(3), atol=1e-5), 'identity composition'\n"
            "assert out3['commutes'] is True, 'two identity rotations commute trivially'\n"
            "\n"
            "# Case 4: random non-commuting pair — SO(3) closure must hold.\n"
            "axis_a = t.tensor([1.0, 2.0, 3.0])\n"
            "axis_b = t.tensor([4.0, -1.0, 0.5])\n"
            "out4 = ex2_compose_and_check(axis_a, 0.9, axis_b, -1.3)\n"
            "assert out4['is_orthogonal'] is True\n"
            "assert out4['det_close_to_one'] is True\n"
            "# R_total @ R_total.T ≈ I (the actual numerical check).\n"
            "I = t.eye(3)\n"
            "assert t.allclose(out4['R_total'] @ out4['R_total'].T, I, atol=1e-5)\n"
            "assert abs(t.linalg.det(out4['R_total']).item() - 1.0) < 1e-5"
        ),
        "solution_body": (
            "def ex2_compose_and_check(axis1, theta1, axis2, theta2):\n"
            "    R1 = rot3d(axis1, theta1)\n"
            "    R2 = rot3d(axis2, theta2)\n"
            "    R_total = R2 @ R1                   # R1 first, then R2\n"
            "    R_swapped = R1 @ R2                 # other order\n"
            "    I = t.eye(3)\n"
            "    is_orthogonal = bool(t.allclose(R_total @ R_total.T, I, atol=1e-5))\n"
            "    det_close_to_one = bool(abs(t.linalg.det(R_total).item() - 1.0) < 1e-5)\n"
            "    commutes = bool(t.allclose(R_total, R_swapped, atol=1e-5))\n"
            "    return {\n"
            "        'R1': R1,\n"
            "        'R2': R2,\n"
            "        'R_total': R_total,\n"
            "        'R_swapped': R_swapped,\n"
            "        'is_orthogonal': is_orthogonal,\n"
            "        'det_close_to_one': det_close_to_one,\n"
            "        'commutes': commutes,\n"
            "    }"
        ),
        "solution_notes": (
            "**Why `R2 @ R1` for 'apply R1 first'.** Conventionally a "
            "rotation acts on a column vector `v` as `R @ v`. To apply "
            "`R1` first, then `R2`, you compute `R2 @ (R1 @ v) = (R2 @ "
            "R1) @ v`. So `R_total = R2 @ R1` matches the 'first then "
            "second' reading order. (Row-vector conventions reverse "
            "this — beware when reading graphics texts.)\n\n"
            "**SO(3) closure.** The set of 3×3 rotation matrices forms "
            "a group under matrix multiplication: closed (product of two "
            "rotations is a rotation), associative, has an identity "
            "(R(0) = I), and every element has an inverse (R(-θ) = "
            "R.T). The closure property is what `is_orthogonal` + "
            "`det_close_to_one` verify numerically.\n\n"
            "**Why non-commutativity matters.** Robotics IK distinguishes "
            "'pitch then yaw' from 'yaw then pitch' — they reach different "
            "poses. Camera control distinguishes 'orbit then tilt' from "
            "'tilt then orbit'. The same axes commute (`R_z(0.4) @ R_z(0.7) "
            "== R_z(0.7) @ R_z(0.4) == R_z(1.1)`), but different axes "
            "do NOT — a fact this drill exposes via the `commutes` flag."
        ),
        "extra_imports": [],
    },

    # ===================================================================
    # segment-line-intersect-2d  —  ex2
    # ex1 = single segment vs single line via linalg.solve. NEW facet:
    # batch N segments vs ONE line, broadcasting the line into the (N, 2, 2)
    # system. Returns t-vector + hit mask. Bloom: Apply.
    # ===================================================================
    {
        "atom_id": "segment-line-intersect-2d",
        "subtopic": "Geometry: Segment-line intersect 2-D",
        "topic_folder": TOPIC_GEOMETRY_CNN,
        "atom_recap_md": RECAP_SEG_LINE,
        "exercise_index": 2,
        "exercise_title": "batched segments vs a single infinite line",
        "slug": "batched-segments-vs-a-single-infinite-line",
        "bloom_level": "Apply",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["2D-geometry", "batched-linalg-solve", "broadcast", "hit-mask"],
        "kcs": [
            "batched-2x2-linear-system",
            "broadcast-line-direction",
        ],
        "lo": (
            "Apply `t.linalg.solve` to a batched `(N, 2, 2)` system to "
            "intersect `N` segments with one infinite line in 2-D, returning "
            "the per-segment t-parameter and a boolean hit mask."
        ),
        "prompt_body": (
            "Implement `ex2_batched_seg_line(S0, S1, L0, L1)`. Find the "
            "intersections of `N` segments with ONE infinite line in 2-D.\n\n"
            "1. `S0`, `S1` are `(N, 2)` float tensors — N segment endpoints. "
            "Segment direction: `d = S1 - S0` (shape `(N, 2)`).\n"
            "2. `L0`, `L1` are `(2,)` float tensors — two points defining the "
            "infinite line. Line direction: `e = L1 - L0` (shape `(2,)`).\n"
            "3. Build the per-segment 2×2 system "
            "`A @ [t, s].T = (L0 - S0)`:\n"
            "   ```python\n"
            "   A = t.stack([d, -e.expand_as(d)], dim=-1)   # (N, 2, 2)\n"
            "   b = L0 - S0                                  # (N, 2)\n"
            "   ts = t.linalg.solve(A, b)                    # (N, 2)\n"
            "   ```\n"
            "   The second column of `A` is `-e` broadcast over `N`.\n"
            "4. Hit mask: `hit = (ts[..., 0] >= 0) & (ts[..., 0] <= 1)` — "
            "closed segment interval, line unconstrained. Shape `(N,)` bool.\n"
            "5. Return `(t_seg, s_line, hit)` where each component has shape "
            "`(N,)`. `t_seg = ts[..., 0]`, `s_line = ts[..., 1]`.\n\n"
            "Assume every 2×2 is non-singular (the parallel case is covered "
            "in the `try-except-solve` atom).\n\n"
            "**Do NOT loop over segments.** The whole point is one batched "
            "`linalg.solve` call.\n\n"
            "Inputs: `S0`, `S1` shape `(N, 2)`; `L0`, `L1` shape `(2,)`.\n"
            "Output: tuple `(t_seg (N,), s_line (N,), hit (N,) bool)`."
        ),
        "stub": (
            "def ex2_batched_seg_line(S0: Tensor, S1: Tensor,\n"
            "                          L0: Tensor, L1: Tensor) -> tuple:\n"
            '    """Intersect N segments with one infinite line; return (t_seg, s_line, hit) all (N,)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Hand-built case: 4 segments vs y=0 line (x-axis through (0,0)-(1,0)).\n"
            "S0 = t.tensor([\n"
            "    [-1.0, -1.0],  # crosses at origin, t=0.5\n"
            "    [-1.0,  1.0],  # crosses at origin, t=0.5\n"
            "    [-1.0,  1.0],  # entirely above x-axis (S1 also above)\n"
            "    [ 0.0,  1.0],  # endpoint S1 on the line (t=1.0)\n"
            "])\n"
            "S1 = t.tensor([\n"
            "    [ 1.0,  1.0],\n"
            "    [ 1.0, -1.0],\n"
            "    [ 2.0,  2.0],\n"
            "    [ 0.0,  0.0],\n"
            "])\n"
            "L0 = t.tensor([0.0, 0.0]); L1 = t.tensor([1.0, 0.0])  # x-axis\n"
            "\n"
            "t_seg, s_line, hit = ex2_batched_seg_line(S0, S1, L0, L1)\n"
            "assert t_seg.shape == (4,), f'expected (4,), got {tuple(t_seg.shape)}'\n"
            "assert s_line.shape == (4,), f'expected (4,), got {tuple(s_line.shape)}'\n"
            "assert hit.shape == (4,), f'expected (4,), got {tuple(hit.shape)}'\n"
            "assert hit.dtype == t.bool, f'hit must be bool, got {hit.dtype}'\n"
            "\n"
            "# Segment 0: crosses at origin, t=0.5.\n"
            "assert abs(t_seg[0].item() - 0.5) < 1e-5, f'seg 0 t: {t_seg[0].item()}'\n"
            "assert hit[0].item() is True\n"
            "# Segment 1: crosses at origin, t=0.5.\n"
            "assert abs(t_seg[1].item() - 0.5) < 1e-5, f'seg 1 t: {t_seg[1].item()}'\n"
            "assert hit[1].item() is True\n"
            "# Segment 2: misses (both endpoints above y=0).\n"
            "assert hit[2].item() is False, f'seg 2 should miss, t={t_seg[2].item()}'\n"
            "assert (t_seg[2].item() < 0.0) or (t_seg[2].item() > 1.0)\n"
            "# Segment 3: endpoint on the line at t=1.0 (closed interval).\n"
            "assert abs(t_seg[3].item() - 1.0) < 1e-5, f'seg 3 t: {t_seg[3].item()}'\n"
            "assert hit[3].item() is True\n"
            "\n"
            "# Cross-check against per-segment loop using a fresh single-segment solver.\n"
            "def _single(S0_i, S1_i, L0_i, L1_i):\n"
            "    d = S1_i - S0_i\n"
            "    e = L1_i - L0_i\n"
            "    A = t.stack([d, -e], dim=1)\n"
            "    b = L0_i - S0_i\n"
            "    ts = t.linalg.solve(A, b)\n"
            "    return ts[0].item(), ts[1].item()\n"
            "\n"
            "for i in range(4):\n"
            "    t_i, s_i = _single(S0[i], S1[i], L0, L1)\n"
            "    assert abs(t_seg[i].item() - t_i) < 1e-5, f'seg {i} batched t disagrees with single-solver: {t_seg[i].item()} vs {t_i}'\n"
            "    assert abs(s_line[i].item() - s_i) < 1e-5, f'seg {i} batched s disagrees: {s_line[i].item()} vs {s_i}'\n"
            "\n"
            "# Stress test: 50 random segments, line is y=2 (horizontal through (0,2),(1,2)).\n"
            "rng = t.Generator().manual_seed(0)\n"
            "N = 50\n"
            "S0r = (t.rand(N, 2, generator=rng) - 0.5) * 8   # in [-4,4]^2\n"
            "S1r = (t.rand(N, 2, generator=rng) - 0.5) * 8\n"
            "L0r = t.tensor([0.0, 2.0]); L1r = t.tensor([1.0, 2.0])\n"
            "t_seg_r, s_line_r, hit_r = ex2_batched_seg_line(S0r, S1r, L0r, L1r)\n"
            "assert t_seg_r.shape == (N,)\n"
            "assert hit_r.shape == (N,)\n"
            "# Hit ↔ segment straddles y=2: i.e. (S0_y - 2) * (S1_y - 2) <= 0.\n"
            "straddle = (S0r[:, 1] - 2.0) * (S1r[:, 1] - 2.0) <= 0\n"
            "assert t.equal(hit_r, straddle), (\n"
            "    f'hit mask disagrees with straddle-y=2 ground truth\\n'\n"
            "    f'hit: {hit_r.tolist()}\\n'\n"
            "    f'straddle: {straddle.tolist()}'\n"
            ")"
        ),
        "solution_body": (
            "def ex2_batched_seg_line(S0, S1, L0, L1):\n"
            "    d = S1 - S0                                       # (N, 2)\n"
            "    e = L1 - L0                                       # (2,)\n"
            "    A = t.stack([d, -e.expand_as(d)], dim=-1)         # (N, 2, 2): cols [d, -e]\n"
            "    b = L0 - S0                                       # (N, 2)\n"
            "    ts = t.linalg.solve(A, b)                         # (N, 2)\n"
            "    t_seg = ts[..., 0]                                # (N,)\n"
            "    s_line = ts[..., 1]                               # (N,)\n"
            "    hit = (t_seg >= 0.0) & (t_seg <= 1.0)             # (N,) bool\n"
            "    return t_seg, s_line, hit"
        ),
        "solution_notes": (
            "**Why `t.stack([..., -e.expand_as(d)], dim=-1)`.** The "
            "matrix `A` has columns `d` and `-e`. With `d: (N, 2)` and "
            "`e: (2,)`, we need to broadcast `-e` along the batch axis "
            "before stacking — `.expand_as(d)` does that without "
            "allocating new memory. `dim=-1` puts the stacked tensors "
            "as COLUMNS (the last axis of the resulting `(N, 2, 2)`). "
            "Stacking on `dim=0` would give `(2, N, 2)` — wrong shape "
            "for `linalg.solve`.\n\n"
            "**Why one batched call beats a Python loop.** PyTorch's "
            "`linalg.solve` uses LAPACK's batched routines internally; "
            "the overhead per system in a batch of 1000 is roughly the "
            "same as one system. A Python loop pays per-iteration Python "
            "+ kernel-launch overhead for every segment. For 1000 "
            "segments the batched form is typically 100-1000x faster.\n\n"
            "**Closed interval `[0, 1]`.** Endpoints on the line count "
            "as hits. The half-open `[0, 1)` convention appears in "
            "raycasting (so consecutive segments don't both claim the "
            "shared endpoint) — different drill. For pure intersection "
            "testing, closed is the canonical choice."
        ),
        "extra_imports": [],
    },
]


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
    print(f"[deepening_t_batch11] Verifying {len(SPECS)} specs...")
    _verify_all(SPECS)

    print(f"\n[deepening_t_batch11] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_t_batch11] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
