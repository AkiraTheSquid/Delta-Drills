#!/usr/bin/env python3
"""Author 8 ex2 deepening drills (batch 12, group Y).

Atoms (3 optimizer-internals + 1 pytorch-modules + 4 resnet-modules):
    - momentum-buffer-update           (ex2: contrast with Nesterov lookahead update)
    - weight-decay-decoupled           (ex2: AdamW vs coupled-Adam-with-L2 → different thetas)
    - weight-decay-l2-add              (ex2: L2-into-Adam ≠ AdamW over 2 steps)
    - dataloader-pin-memory-workers    (ex2: worker_init_fn seeded shuffle preserves batch shape)
    - 1x1-conv-channel-reshape         (ex2: 1x1 conv preserves rectangular H,W spatial dims)
    - batchnorm-running-stats          (ex2: eval mode uses running stats, NOT batch stats)
    - maxpool-reduce                   (ex2: contrast max vs avg via einops.reduce)
    - register-buffer                  (ex2: register_buffer vs nn.Parameter — state_dict vs parameters)

Each ex2 hits a DISTINCT facet from ex1. ONE LO + ONE Bloom + <=2 KCs per drill.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC_OPT = "prereqs_optimizer_internals"
TOPIC_PT = "prereqs_pytorch_modules"
TOPIC_RES = "prereqs_resnet_modules"


# ---------------------------------------------------------------------------
# Recap blocks
# ---------------------------------------------------------------------------

RECAP_NESTEROV = (
    "## Nesterov momentum — evaluate g at the lookahead point\n"
    "\n"
    "Ex1 implemented the classical heavy-ball momentum update:\n"
    "```\n"
    "b ← μ·b + g            # in-place momentum buffer update\n"
    "θ ← θ − lr·b           # parameter step\n"
    "```\n"
    "\n"
    "Nesterov accelerated gradient (NAG) changes ONE thing: the gradient `g` "
    "is evaluated at the LOOKAHEAD point `θ − lr·μ·b` instead of at the "
    "current `θ`. PyTorch's `SGD(nesterov=True)` collapses this into the same "
    "buffer machinery via:\n"
    "```\n"
    "b ← μ·b + g            # same buffer update\n"
    "θ ← θ − lr·(g + μ·b)   # step uses g + μ·b instead of just b\n"
    "```\n"
    "\n"
    "**Why the `g + μ·b` form is equivalent.** Expanding the lookahead "
    "gradient via the chain rule and reorganising terms gives back exactly "
    "`g + μ·b` as the effective descent direction — without ever evaluating "
    "the gradient at a second point. Same FLOPs as plain momentum, one extra "
    "tensor add.\n"
    "\n"
    "**Same in-place buffer contract.** The buffer list is still mutated in "
    "place (ex1's invariant). Only the formula for the per-step update "
    "direction changes."
)

RECAP_ADAMW_VS_COUPLED = (
    "## Decoupled (AdamW) vs coupled (Adam+L2) — different θ after one step\n"
    "\n"
    "Ex1 implemented one AdamW step where weight decay is applied DIRECTLY to "
    "θ, decoupled from the adaptive gradient:\n"
    "```\n"
    "# AdamW (decoupled):\n"
    "m ← β1·m + (1−β1)·g\n"
    "v ← β2·v + (1−β2)·g²\n"
    "m̂ = m / (1 − β1^t)\n"
    "v̂ = v / (1 − β2^t)\n"
    "θ ← θ − lr·(m̂ / (√v̂ + ε)) − lr·wd·θ\n"
    "```\n"
    "\n"
    "Coupled Adam-with-L2 (the BUGGY pre-AdamW formulation) folds wd·θ into "
    "the gradient FIRST, then runs the adaptive Adam update on that augmented "
    "gradient:\n"
    "```\n"
    "# Coupled Adam + L2:\n"
    "g' = g + wd·θ\n"
    "m ← β1·m + (1−β1)·g'\n"
    "v ← β2·v + (1−β2)·g'²\n"
    "m̂ = m / (1 − β1^t)\n"
    "v̂ = v / (1 − β2^t)\n"
    "θ ← θ − lr·(m̂ / (√v̂ + ε))\n"
    "```\n"
    "\n"
    "**The two are NOT equivalent.** Because Adam normalises by `√v̂`, "
    "folding `wd·θ` into `g` makes the decay's effective magnitude depend on "
    "the second-moment estimate — small-gradient parameters get DECAYED LESS "
    "than they should. Decoupling fixes this. Two implementations on the "
    "SAME `g`, `m`, `v`, `θ` will produce DIFFERENT `θ_new`.\n"
    "\n"
    "**Loshchilov & Hutter (2019)** showed this is why \"Adam with weight "
    "decay\" generalised worse than SGD+L2: the two updates the world thought "
    "were the same were not."
)

RECAP_L2_INTO_ADAM = (
    "## L2-into-gradient over 2 steps — AdamW divergence visible quickly\n"
    "\n"
    "Ex1 folded `λ·θ` into `g` — the canonical L2 regularisation move. This "
    "is fine for plain SGD (it's exactly equivalent to penalising "
    "½λ‖θ‖² in the loss). The deepening move shows where it BREAKS: stacked "
    "with Adam's adaptive normalisation, the L2 path and the decoupled "
    "(AdamW) path diverge after just 2 steps on a single scalar parameter.\n"
    "\n"
    "**The experiment.** Same θ₀, same g₁, same g₂, same hparams. Two "
    "trajectories:\n"
    "- **L2 path:** at each step, compute `g' = g + λ·θ`, run Adam on `g'`.\n"
    "- **AdamW path:** at each step, run Adam on `g`, then `θ ← θ − lr·λ·θ`.\n"
    "\n"
    "After 2 steps, `θ_L2 ≠ θ_AdamW`. The gap grows with `v̂`'s "
    "asymmetry across steps — small-gradient steps amplify the decoupling "
    "difference.\n"
    "\n"
    "**Why this matters for ex1.** `apply_weight_decay` is the right tool "
    "for SGD-style optimisers; chaining it INTO Adam's gradient stream "
    "silently couples decay to the adaptive scale. The fix isn't to fix L2 — "
    "it's to switch to AdamW."
)

RECAP_DL_WORKER_SEED = (
    "## DataLoader + `worker_init_fn` + shuffle — batch shape stays correct\n"
    "\n"
    "Ex1 built a DataLoader with `num_workers` and `pin_memory` configured. "
    "The deepening move shows that under `shuffle=True` with multiple "
    "workers, batch shapes (and batch counts) remain correct as long as the "
    "main-process seed is set AND each worker gets a deterministic seed via "
    "`worker_init_fn`.\n"
    "\n"
    "```python\n"
    "def worker_init_fn(worker_id):\n"
    "    base = torch.initial_seed() % (2**32)\n"
    "    np.random.seed(base + worker_id)\n"
    "    random.seed(base + worker_id)\n"
    "```\n"
    "\n"
    "**Why this matters.** `torch.initial_seed()` inside a worker returns "
    "the per-worker base seed PyTorch assigns. Without re-seeding numpy and "
    "Python `random`, every worker draws the same numpy/random stream — a "
    "silent source of duplicate augmentations in data-augmentation pipelines.\n"
    "\n"
    "**Shape invariant.** Regardless of shuffle / workers / pin_memory, the "
    "DataLoader still yields `(batch_size, *item_shape)` tensors and the "
    "total number of items emitted across one epoch equals `len(dataset)`. "
    "`pin_memory=False` on CPU; the shape contract is unchanged.\n"
    "\n"
    "**CPU-only is fine.** `pin_memory=True` requires CUDA at runtime. In a "
    "CPU test we pass `pin_memory=False` and still exercise the rest of the "
    "config (workers, shuffle, seeded init)."
)

RECAP_CONV_SPATIAL = (
    "## 1×1 conv preserves H, W — even for rectangular inputs\n"
    "\n"
    "Ex1 showed that a 1×1 `Conv2d` is per-pixel linear: it remixes the "
    "channel axis but leaves the spatial axes UNTOUCHED. The deepening move "
    "verifies this on a rectangular `H ≠ W` input — both axes pass through "
    "unchanged.\n"
    "\n"
    "```\n"
    "Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0)\n"
    "input:  (B, C_in,  H, W)\n"
    "output: (B, C_out, H, W)     # H, W identical to input\n"
    "```\n"
    "\n"
    "**Equivalence to per-pixel `nn.Linear`.** Flatten the spatial axes to "
    "`(B·H·W, C_in)`, apply `Linear(C_in, C_out)` with the conv's weight + "
    "bias reshaped, then un-flatten back to `(B, C_out, H, W)`. The two "
    "outputs match to machine precision.\n"
    "\n"
    "**Why rectangular inputs are the load-bearing test.** A 1×1 conv on a "
    "square input could pass a misimplemented 'transpose H/W' bug "
    "accidentally. A non-square `(H=7, W=11)` input would FAIL if the "
    "implementation mixed H and W up. ARENA's vision pipelines see "
    "rectangular feature maps after asymmetric pooling — this is not a toy."
)

RECAP_BN_EVAL = (
    "## BatchNorm `.eval()` uses running stats, not batch stats\n"
    "\n"
    "Ex1 updated `running_mean` and `running_var` via the EMA in train mode "
    "only. The deepening move toggles `.train()` vs `.eval()` on the same "
    "input and observes that the OUTPUT itself changes — eval mode uses the "
    "FIXED running stats, while train mode uses the batch's own mean/var.\n"
    "\n"
    "```\n"
    "train mode:  y = (x − batch_mean) / √(batch_var + ε) · γ + β\n"
    "             [running_mean, running_var update via EMA]\n"
    "eval  mode:  y = (x − running_mean) / √(running_var + ε) · γ + β\n"
    "             [no update to running stats]\n"
    "```\n"
    "\n"
    "**Why this matters at inference.** A model trained with BN sees batch "
    "stats during training; at deployment the batch may be size 1 (single "
    "image), and batch stats become meaningless or undefined (var = 0 for "
    "B=1). Eval mode swaps to the running stats accumulated over training, "
    "giving stable, batch-independent outputs.\n"
    "\n"
    "**Same module, two outputs.** Calling `bn.eval()` then `bn.train()` on "
    "the same `bn` and feeding the same `x` returns DIFFERENT tensors. The "
    "difference is the canonical 'why does my model behave weirdly at "
    "inference' debugging step for any new BN user."
)

RECAP_MAX_VS_AVG = (
    "## MaxPool vs AvgPool — same reduce shape, different reduction op\n"
    "\n"
    "Ex1 built `MaxPool2d` via `einops.reduce(x, '... (h p1) (w p2) -> ... h "
    "w', 'max', p1=p, p2=p)`. The deepening move shows that swapping ONE "
    "string — `'max'` → `'mean'` — gives `AvgPool2d` over the same window. "
    "Same window geometry, same output shape, different aggregation.\n"
    "\n"
    "```python\n"
    "max_out  = reduce(x, '... (h p1) (w p2) -> ... h w', 'max',  p1=p, p2=p)\n"
    "avg_out  = reduce(x, '... (h p1) (w p2) -> ... h w', 'mean', p1=p, p2=p)\n"
    "```\n"
    "\n"
    "**Shape invariant.** Both outputs have shape `(..., H/p, W/p)`. The "
    "reduction op is orthogonal to the einops pattern.\n"
    "\n"
    "**Why these are NOT the same.** Max preserves outliers (a single "
    "bright pixel survives the pool); average dilutes them by `p²`. For "
    "ReLU activation maps where most entries are zero, max-pool keeps a "
    "sparse signal alive; avg-pool kills it. This is why classification "
    "backbones use max-pool and segmentation decoders use avg-pool."
)

RECAP_BUF_VS_PARAM = (
    "## `register_buffer` vs `nn.Parameter` — state_dict yes, parameters no\n"
    "\n"
    "Ex1 registered non-trainable state via `register_buffer` and verified "
    "it appears in `state_dict()`. The deepening move pins the CONTRAST "
    "against `nn.Parameter`:\n"
    "\n"
    "| trait                              | `register_buffer` | `nn.Parameter` |\n"
    "|------------------------------------|-------------------|----------------|\n"
    "| appears in `state_dict()`          | yes               | yes            |\n"
    "| appears in `.parameters()`         | NO                | yes            |\n"
    "| receives gradients                 | NO                | yes (if `requires_grad`)|\n"
    "| moves with `.to(device)`           | yes               | yes            |\n"
    "| saved on `torch.save(model)`       | yes               | yes            |\n"
    "\n"
    "**Why both are needed.** BN's `running_mean` MUST be saved (state_dict) "
    "AND must travel with the model to GPU (`.to(device)`) — but it MUST NOT "
    "be optimised by `torch.optim` (no gradient). Buffers exist exactly for "
    "this 'persistent but non-trainable' slot.\n"
    "\n"
    "**The diagnostic.** `len(list(module.parameters()))` should be 0 for a "
    "pure-buffer module; `len(module.state_dict())` should equal the number "
    "of registered buffers."
)


# ---------------------------------------------------------------------------
# SPEC 1 — momentum-buffer-update ex2 (Nesterov)
# ---------------------------------------------------------------------------

SPEC_NESTEROV = {
    "atom_id": "momentum-buffer-update",
    "subtopic": "Optimizer: Momentum buffer",
    "topic_folder": TOPIC_OPT,
    "atom_recap_md": RECAP_NESTEROV,
    "exercise_index": 2,
    "exercise_title": "Nesterov momentum step using the buffer + (g + μ·b) form",
    "slug": "nesterov-momentum-step-via-buffer-plus-lookahead",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["nesterov", "momentum", "lookahead", "optimizer"],
    "kcs": [
        "in-place-buffer-update",
        "nesterov-effective-direction-g-plus-mu-b",
    ],
    "lo": (
        "Apply PyTorch's Nesterov-via-buffer formulation: update each "
        "buffer in place as `b ← μ·b + g`, then compute the per-parameter "
        "descent direction as `g + μ·b` — distinct from classical "
        "momentum's plain `b`."
    ),
    "prompt_body": (
        "Implement `ex2_nesterov_step(buffer_list, grad_list, mu)`. The "
        "Nesterov deepening of ex1's classical momentum.\n\n"
        "Contract:\n\n"
        "1. Mutate each `buffer_list[i]` IN PLACE: `b ← μ·b + g`. Same "
        "in-place rule as ex1 — do NOT replace the buffer object.\n"
        "2. Return a NEW list of per-parameter descent directions: "
        "`d_i = g_i + μ·b_i` (after the buffer update).\n"
        "3. `grad_list` is read-only — do NOT mutate the gradient tensors.\n"
        "4. Lengths match: `len(buffer_list) == len(grad_list)`. (Caller "
        "guarantees this; you don't need to validate.)\n\n"
        "Inputs:\n"
        "- `buffer_list`: `list[Tensor]`, mutated in place.\n"
        "- `grad_list`: `list[Tensor]`, read-only.\n"
        "- `mu`: `float`, momentum coefficient.\n\n"
        "Output: `list[Tensor]` of descent directions (same length as "
        "inputs). The CALLER applies `θ ← θ − lr·d`."
    ),
    "stub": (
        "def ex2_nesterov_step(buffer_list: list, grad_list: list, mu: float) -> list:\n"
        '    """Update buffers in place, then return [g + μ·b] descent directions."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Single-tensor Nesterov ===\n"
        "b = t.tensor([1.0, 2.0])\n"
        "g = t.tensor([0.5, -1.0])\n"
        "b_id_before = id(b)\n"
        "out = ex2_nesterov_step([b], [g], mu=0.9)\n"
        "# Buffer mutated in place:\n"
        "assert id(b) == b_id_before, 'buffer object must be same instance after call'\n"
        "expected_b = t.tensor([0.9 * 1.0 + 0.5, 0.9 * 2.0 + (-1.0)])\n"
        "assert t.allclose(b, expected_b), f'buffer wrong: {b} vs {expected_b}'\n"
        "# Descent direction = g + μ·b (post-update):\n"
        "expected_d = g + 0.9 * b\n"
        "assert t.allclose(out[0], expected_d), f'descent dir wrong: {out[0]} vs {expected_d}'\n"
        "\n"
        "# === Grad list NOT mutated ===\n"
        "g_orig = g.clone()\n"
        "b2 = t.zeros(2)\n"
        "ex2_nesterov_step([b2], [g], mu=0.9)\n"
        "assert t.allclose(g, g_orig), 'grad_list must not be mutated'\n"
        "\n"
        "# === Two-tensor list keeps order + in-place semantics for each ===\n"
        "b_a = t.tensor([0.0, 0.0])\n"
        "b_b = t.tensor([1.0, -1.0])\n"
        "g_a = t.tensor([2.0, 3.0])\n"
        "g_b = t.tensor([-0.5, 0.5])\n"
        "ids = [id(b_a), id(b_b)]\n"
        "out = ex2_nesterov_step([b_a, b_b], [g_a, g_b], mu=0.5)\n"
        "assert [id(b_a), id(b_b)] == ids, 'all buffers must stay same instances'\n"
        "assert t.allclose(b_a, 0.5 * t.zeros(2) + t.tensor([2.0, 3.0]))\n"
        "assert t.allclose(b_b, 0.5 * t.tensor([1.0, -1.0]) + t.tensor([-0.5, 0.5]))\n"
        "assert t.allclose(out[0], g_a + 0.5 * b_a)\n"
        "assert t.allclose(out[1], g_b + 0.5 * b_b)\n"
        "assert len(out) == 2\n"
        "\n"
        "# === mu=0 → buffer becomes g, descent = g (no momentum component) ===\n"
        "b = t.tensor([5.0, -5.0])\n"
        "g = t.tensor([1.0, 1.0])\n"
        "out = ex2_nesterov_step([b], [g], mu=0.0)\n"
        "assert t.allclose(b, g), 'mu=0: buffer should equal g after update'\n"
        "assert t.allclose(out[0], g), 'mu=0: descent direction = g + 0·b = g'\n"
        "\n"
        "# === Nesterov ≠ classical (the WHOLE POINT of this drill) ===\n"
        "# Classical: descent = b (post-update). Nesterov: descent = g + μ·b.\n"
        "b = t.tensor([1.0])\n"
        "g = t.tensor([2.0])\n"
        "mu = 0.9\n"
        "out = ex2_nesterov_step([b], [g], mu=mu)\n"
        "# Classical descent would be b post-update = 0.9*1 + 2 = 2.9.\n"
        "# Nesterov descent = g + μ·b = 2.0 + 0.9 * 2.9 = 4.61.\n"
        "assert t.allclose(out[0], t.tensor([4.61])), f'Nesterov direction must differ from classical, got {out[0]}'\n"
        "\n"
        "# === Empty list → empty output ===\n"
        "out = ex2_nesterov_step([], [], mu=0.9)\n"
        "assert out == [], f'empty input → empty output, got {out}'\n"
        "\n"
        "# === Output is a list of Tensors, not a generator ===\n"
        "out = ex2_nesterov_step([t.zeros(3)], [t.ones(3)], mu=0.5)\n"
        "assert isinstance(out, list), f'output must be list, got {type(out).__name__}'\n"
        "assert all(isinstance(x, t.Tensor) for x in out), 'each element must be a Tensor'"
    ),
    "solution_body": (
        "def ex2_nesterov_step(buffer_list, grad_list, mu):\n"
        "    out = []\n"
        "    for b, g in zip(buffer_list, grad_list):\n"
        "        # In-place buffer update: b ← μ·b + g\n"
        "        b.mul_(mu).add_(g)\n"
        "        # Nesterov descent direction: g + μ·b (post-update).\n"
        "        out.append(g + mu * b)\n"
        "    return out"
    ),
    "solution_notes": (
        "**`b.mul_(mu).add_(g)` is the in-place pattern.** Chains two "
        "in-place ops on the same storage. `b = mu * b + g` would allocate "
        "a new tensor and lose the in-place contract — the caller's external "
        "reference would still point at the OLD buffer.\n\n"
        "**Why `g + μ·b` after the buffer update.** The classical descent "
        "direction is just `b` (which equals `μ·b_old + g`). The Nesterov "
        "form adds another `μ·b_post` on top — equivalent to evaluating the "
        "gradient at the lookahead point WITHOUT actually doing two forward "
        "passes.\n\n"
        "**Grad never mutated.** The Nesterov formulation does NOT rewrite "
        "`g`. The caller can keep using the original gradient tensor for "
        "logging or gradient clipping after this call."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 2 — weight-decay-decoupled ex2 (AdamW vs coupled)
# ---------------------------------------------------------------------------

SPEC_ADAMW_VS_COUPLED = {
    "atom_id": "weight-decay-decoupled",
    "subtopic": "Optimizer: decoupled weight decay (AdamW)",
    "topic_folder": TOPIC_OPT,
    "atom_recap_md": RECAP_ADAMW_VS_COUPLED,
    "exercise_index": 2,
    "exercise_title": "compare one AdamW step against one coupled-Adam-with-L2 step",
    "slug": "compare-adamw-vs-coupled-adam-l2-one-step",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["adamw", "weight-decay", "coupled", "decoupled"],
    "kcs": [
        "decoupled-decay-direct-on-theta",
        "coupled-decay-folded-into-gradient",
    ],
    "lo": (
        "Analyse the difference between decoupled (AdamW) and coupled "
        "(Adam+L2) weight-decay by implementing BOTH one-step updates on "
        "the same inputs and showing the resulting parameters differ — "
        "because Adam's `√v̂` normalisation re-scales decay only in the "
        "coupled path."
    ),
    "prompt_body": (
        "Implement `ex2_compare_decay_one_step(p, grad, m, v, lr, beta1, "
        "beta2, eps, wd, step)`. Return a tuple `(p_adamw, p_coupled)` of "
        "the two updated parameters.\n\n"
        "Both paths share inputs (`p`, `grad`, `m`, `v`, hparams, `step`). "
        "Treat `m`, `v` as immutable from the caller's perspective: do NOT "
        "mutate them — clone inside if needed.\n\n"
        "**AdamW path (decoupled):**\n"
        "```\n"
        "m_aw ← β1·m + (1−β1)·grad\n"
        "v_aw ← β2·v + (1−β2)·grad²\n"
        "m̂_aw = m_aw / (1 − β1**step)\n"
        "v̂_aw = v_aw / (1 − β2**step)\n"
        "p_adamw = p − lr · (m̂_aw / (√v̂_aw + eps))  −  lr · wd · p\n"
        "```\n\n"
        "**Coupled path (Adam + L2):**\n"
        "```\n"
        "g' = grad + wd · p\n"
        "m_cp ← β1·m + (1−β1)·g'\n"
        "v_cp ← β2·v + (1−β2)·g'²\n"
        "m̂_cp = m_cp / (1 − β1**step)\n"
        "v̂_cp = v_cp / (1 − β2**step)\n"
        "p_coupled = p − lr · (m̂_cp / (√v̂_cp + eps))\n"
        "```\n\n"
        "Inputs: `p`, `grad`, `m`, `v` are all `Tensor`s of the same shape. "
        "`step` is 1-indexed (matches PyTorch's Adam state).\n\n"
        "Output: tuple `(p_adamw, p_coupled)` — both fresh tensors. "
        "Original `p` is unchanged."
    ),
    "stub": (
        "def ex2_compare_decay_one_step(p, grad, m, v, lr, beta1, beta2, eps, wd, step):\n"
        '    """Return (p_adamw, p_coupled) — both one-step updates on the same inputs."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import math\n"
        "\n"
        "# === Both updates run, return fresh tensors ===\n"
        "p = t.tensor([0.1, 0.2, -0.3])\n"
        "grad = t.tensor([0.05, -0.02, 0.04])\n"
        "m = t.zeros(3)\n"
        "v = t.zeros(3)\n"
        "p_aw, p_cp = ex2_compare_decay_one_step(p, grad, m, v, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.1, step=1)\n"
        "assert p_aw.shape == p.shape and p_cp.shape == p.shape\n"
        "assert id(p_aw) != id(p) and id(p_cp) != id(p), 'must return fresh tensors'\n"
        "\n"
        "# === The two updates differ — the whole point ===\n"
        "assert not t.allclose(p_aw, p_cp, atol=1e-7), (\n"
        "    f'AdamW and coupled-Adam-L2 must yield different params; got identical: {p_aw} vs {p_cp}'\n"
        ")\n"
        "\n"
        "# === Inputs unchanged ===\n"
        "assert t.allclose(p, t.tensor([0.1, 0.2, -0.3])), 'p must not be mutated'\n"
        "assert t.allclose(m, t.zeros(3)), 'm must not be mutated'\n"
        "assert t.allclose(v, t.zeros(3)), 'v must not be mutated'\n"
        "\n"
        "# === wd=0 → both paths coincide (decay vanishes in both) ===\n"
        "p_aw0, p_cp0 = ex2_compare_decay_one_step(p, grad, m, v, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0, step=1)\n"
        "assert t.allclose(p_aw0, p_cp0, atol=1e-7), f'wd=0 must give identical updates, got {p_aw0} vs {p_cp0}'\n"
        "\n"
        "# === Hand-verify AdamW path on a scalar ===\n"
        "p_s = t.tensor([1.0])\n"
        "g_s = t.tensor([0.5])\n"
        "m_s = t.tensor([0.0])\n"
        "v_s = t.tensor([0.0])\n"
        "p_aw_s, _ = ex2_compare_decay_one_step(p_s, g_s, m_s, v_s, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01, step=1)\n"
        "# m1 = 0.1*0.5 = 0.05; v1 = 0.001*0.25 = 0.00025\n"
        "# m̂ = 0.05/(1-0.9) = 0.5; v̂ = 0.00025/(1-0.999) = 0.25\n"
        "# adaptive step = 0.5 / (sqrt(0.25) + 1e-8) ≈ 1.0\n"
        "# p_aw = 1.0 - 0.1*1.0 - 0.1*0.01*1.0 = 1.0 - 0.1 - 0.001 = 0.899\n"
        "assert t.allclose(p_aw_s, t.tensor([0.899]), atol=1e-4), f'AdamW scalar wrong: {p_aw_s}'\n"
        "\n"
        "# === Hand-verify coupled path on the same scalar ===\n"
        "_, p_cp_s = ex2_compare_decay_one_step(p_s, g_s, m_s, v_s, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01, step=1)\n"
        "# g' = 0.5 + 0.01 * 1.0 = 0.51\n"
        "# m = 0.1*0.51 = 0.051; v = 0.001*0.2601 = 0.0002601\n"
        "# m̂ = 0.051/0.1 = 0.51; v̂ = 0.0002601/0.001 = 0.2601\n"
        "# adaptive step = 0.51 / (sqrt(0.2601) + 1e-8) = 0.51 / 0.51 = 1.0\n"
        "# p_cp = 1.0 - 0.1*1.0 = 0.9\n"
        "assert t.allclose(p_cp_s, t.tensor([0.9]), atol=1e-4), f'coupled scalar wrong: {p_cp_s}'\n"
        "# Compared: AdamW ≈ 0.899, coupled ≈ 0.9. They differ.\n"
        "assert not t.allclose(p_aw_s, p_cp_s, atol=1e-7)\n"
        "\n"
        "# === Step 2: increment of state. Differences ACCUMULATE. ===\n"
        "p2 = t.tensor([0.5, -0.5])\n"
        "g2 = t.tensor([0.1, 0.1])\n"
        "m_warm = t.tensor([0.02, 0.03])\n"
        "v_warm = t.tensor([0.001, 0.002])\n"
        "p_aw2, p_cp2 = ex2_compare_decay_one_step(p2, g2, m_warm, v_warm, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.05, step=2)\n"
        "diff = (p_aw2 - p_cp2).abs()\n"
        "assert diff.max() > 1e-5, f'AdamW vs coupled must visibly differ at step 2, got diff max {diff.max().item()}'\n"
        "\n"
        "# === Return type is a 2-tuple of Tensors ===\n"
        "ret = ex2_compare_decay_one_step(p, grad, m, v, 1e-2, 0.9, 0.999, 1e-8, 0.1, 1)\n"
        "assert isinstance(ret, tuple) and len(ret) == 2\n"
        "assert isinstance(ret[0], t.Tensor) and isinstance(ret[1], t.Tensor)"
    ),
    "solution_body": (
        "def ex2_compare_decay_one_step(p, grad, m, v, lr, beta1, beta2, eps, wd, step):\n"
        "    # --- AdamW path: decoupled decay applied to p directly ---\n"
        "    m_aw = beta1 * m + (1 - beta1) * grad\n"
        "    v_aw = beta2 * v + (1 - beta2) * grad * grad\n"
        "    m_hat_aw = m_aw / (1 - beta1 ** step)\n"
        "    v_hat_aw = v_aw / (1 - beta2 ** step)\n"
        "    p_adamw = p - lr * (m_hat_aw / (v_hat_aw.sqrt() + eps)) - lr * wd * p\n"
        "\n"
        "    # --- Coupled path: fold wd·p into gradient, then run plain Adam ---\n"
        "    g_eff = grad + wd * p\n"
        "    m_cp = beta1 * m + (1 - beta1) * g_eff\n"
        "    v_cp = beta2 * v + (1 - beta2) * g_eff * g_eff\n"
        "    m_hat_cp = m_cp / (1 - beta1 ** step)\n"
        "    v_hat_cp = v_cp / (1 - beta2 ** step)\n"
        "    p_coupled = p - lr * (m_hat_cp / (v_hat_cp.sqrt() + eps))\n"
        "\n"
        "    return p_adamw, p_coupled"
    ),
    "solution_notes": (
        "**Why the wd=0 case must coincide.** When `wd=0`, the AdamW decay "
        "term vanishes; the coupled path's augmented gradient `g + 0·p = g` "
        "is the same as plain Adam. Both updates collapse to vanilla Adam. "
        "This is the boundary test — if your implementation disagrees here, "
        "you have a bug in the BASE update, not in the decay handling.\n\n"
        "**Why `v.sqrt() + eps` not `(v + eps).sqrt()`.** PyTorch's Adam uses "
        "the former (`addcdiv_` form). For non-pathological `v̂`, both round "
        "to the same value. PyTorch tests against the official Adam paper's "
        "formulation; matching it makes regression tests cleaner.\n\n"
        "**The scalar handwork.** Step 1, scalar p, m=v=0: the bias-corrected "
        "first/second moment perfectly cancel each other's normalisation "
        "(`m̂/√v̂ = sign(g)`), so the adaptive step is exactly 1. This makes "
        "the WHOLE update reducible to mental arithmetic — a useful sanity "
        "check pattern for any Adam variant."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 3 — weight-decay-l2-add ex2 (L2-into-Adam ≠ AdamW over 2 steps)
# ---------------------------------------------------------------------------

SPEC_L2_INTO_ADAM = {
    "atom_id": "weight-decay-l2-add",
    "subtopic": "Optimizer: Weight decay L2",
    "topic_folder": TOPIC_OPT,
    "atom_recap_md": RECAP_L2_INTO_ADAM,
    "exercise_index": 2,
    "exercise_title": "run 2 Adam-steps with L2-into-grad and AdamW side by side; verify divergence",
    "slug": "two-step-l2-into-adam-vs-adamw-divergence",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["l2", "adam", "adamw", "two-step"],
    "kcs": [
        "l2-fold-into-gradient",
        "two-step-state-accumulation",
    ],
    "lo": (
        "Analyse where L2-into-gradient (ex1's `apply_weight_decay`) breaks "
        "when chained with Adam: simulate two Adam steps along the L2 path "
        "and along the decoupled (AdamW) path and verify the resulting "
        "θ trajectories diverge."
    ),
    "prompt_body": (
        "Implement `ex2_two_step_l2_vs_adamw(theta0, grads, lr, beta1, "
        "beta2, eps, lmda)`.\n\n"
        "`grads` is a `list[Tensor]` of length 2 — the gradients at step 1 "
        "and step 2 (assumed already computed by the caller — no autograd "
        "needed).\n\n"
        "Run TWO trajectories of TWO steps each, both starting from the "
        "same `theta0` with `m = v = 0`. Steps are 1-indexed.\n\n"
        "**L2 path** (ex1's `apply_weight_decay` folded into Adam's grad):\n"
        "```\n"
        "for step in (1, 2):\n"
        "    g = grads[step-1] + lmda * theta\n"
        "    m ← beta1·m + (1-beta1)·g\n"
        "    v ← beta2·v + (1-beta2)·g²\n"
        "    m̂ = m / (1 - beta1**step)\n"
        "    v̂ = v / (1 - beta2**step)\n"
        "    theta ← theta - lr · (m̂ / (√v̂ + eps))\n"
        "```\n\n"
        "**AdamW path** (decoupled):\n"
        "```\n"
        "for step in (1, 2):\n"
        "    g = grads[step-1]                    # no fold\n"
        "    m ← beta1·m + (1-beta1)·g\n"
        "    v ← beta2·v + (1-beta2)·g²\n"
        "    m̂ = m / (1 - beta1**step)\n"
        "    v̂ = v / (1 - beta2**step)\n"
        "    theta ← theta - lr · (m̂ / (√v̂ + eps)) - lr · lmda · theta\n"
        "```\n\n"
        "Return tuple `(theta_l2_final, theta_adamw_final)` after both "
        "trajectories complete 2 steps."
    ),
    "stub": (
        "def ex2_two_step_l2_vs_adamw(theta0, grads, lr, beta1, beta2, eps, lmda):\n"
        '    """Run 2 steps of L2-path-Adam and AdamW from the same θ₀; return (θ_l2, θ_adamw)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Two trajectories complete, return fresh tensors ===\n"
        "theta0 = t.tensor([1.0, -1.0])\n"
        "grads = [t.tensor([0.1, -0.05]), t.tensor([0.08, -0.03])]\n"
        "th_l2, th_aw = ex2_two_step_l2_vs_adamw(theta0, grads, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, lmda=0.1)\n"
        "assert th_l2.shape == theta0.shape\n"
        "assert th_aw.shape == theta0.shape\n"
        "\n"
        "# === The two trajectories diverge after 2 steps (the WHOLE POINT) ===\n"
        "diff = (th_l2 - th_aw).abs()\n"
        "assert diff.max() > 1e-6, (\n"
        "    f'L2-into-Adam and AdamW must yield different θ after 2 steps; '\n"
        "    f'got identical: {th_l2} vs {th_aw}'\n"
        ")\n"
        "\n"
        "# === theta0 unchanged (no mutation contract) ===\n"
        "assert t.allclose(theta0, t.tensor([1.0, -1.0])), 'theta0 must not be mutated'\n"
        "\n"
        "# === lmda=0 → both paths coincide ===\n"
        "th_l2_0, th_aw_0 = ex2_two_step_l2_vs_adamw(theta0, grads, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, lmda=0.0)\n"
        "assert t.allclose(th_l2_0, th_aw_0, atol=1e-7), f'lmda=0: paths must coincide, got {th_l2_0} vs {th_aw_0}'\n"
        "\n"
        "# === Hand-verify one step of L2 path on a scalar ===\n"
        "th = t.tensor([1.0])\n"
        "gs = [t.tensor([0.5]), t.tensor([0.5])]\n"
        "th_l2_s, _ = ex2_two_step_l2_vs_adamw(th, gs, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, lmda=0.01)\n"
        "# Step 1: g' = 0.5 + 0.01*1.0 = 0.51; m=0.051; v=0.0002601\n"
        "#         m̂=0.51, v̂=0.2601, step = 0.51/sqrt(0.2601) ≈ 1.0\n"
        "#         θ_1 = 1.0 - 0.1*1.0 = 0.9\n"
        "# Step 2: g' = 0.5 + 0.01*0.9 = 0.509; carry m, v forward.\n"
        "# We don't hand-compute step 2 exactly; just verify it ran (changed).\n"
        "assert th_l2_s.item() < 0.9, f'L2 path must take a 2nd step too, got {th_l2_s.item()}'\n"
        "\n"
        "# === Both paths apply decay (both move closer to 0 than no-decay would) ===\n"
        "th_no_decay, _ = ex2_two_step_l2_vs_adamw(theta0, grads, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, lmda=0.0)\n"
        "th_with_decay_l2, th_with_decay_aw = ex2_two_step_l2_vs_adamw(theta0, grads, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, lmda=0.5)\n"
        "# Decay should shrink positive components and shrink-magnitude negative components.\n"
        "assert th_with_decay_l2[0].abs() < th_no_decay[0].abs() or th_with_decay_aw[0].abs() < th_no_decay[0].abs()\n"
        "\n"
        "# === Return type is a 2-tuple of Tensors ===\n"
        "ret = ex2_two_step_l2_vs_adamw(theta0, grads, 1e-2, 0.9, 0.999, 1e-8, 0.1)\n"
        "assert isinstance(ret, tuple) and len(ret) == 2\n"
        "assert isinstance(ret[0], t.Tensor) and isinstance(ret[1], t.Tensor)\n"
        "\n"
        "# === Larger lmda → bigger divergence between the two paths ===\n"
        "th_l2_a, th_aw_a = ex2_two_step_l2_vs_adamw(theta0, grads, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, lmda=0.05)\n"
        "th_l2_b, th_aw_b = ex2_two_step_l2_vs_adamw(theta0, grads, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, lmda=0.5)\n"
        "gap_small = (th_l2_a - th_aw_a).abs().max().item()\n"
        "gap_large = (th_l2_b - th_aw_b).abs().max().item()\n"
        "assert gap_large > gap_small, f'larger lmda must produce larger gap; got {gap_large} not > {gap_small}'"
    ),
    "solution_body": (
        "def ex2_two_step_l2_vs_adamw(theta0, grads, lr, beta1, beta2, eps, lmda):\n"
        "    assert len(grads) == 2\n"
        "\n"
        "    # --- L2 path: fold lmda·theta into the gradient stream ---\n"
        "    theta = theta0.clone()\n"
        "    m = t.zeros_like(theta)\n"
        "    v = t.zeros_like(theta)\n"
        "    for step in (1, 2):\n"
        "        g = grads[step - 1] + lmda * theta\n"
        "        m = beta1 * m + (1 - beta1) * g\n"
        "        v = beta2 * v + (1 - beta2) * g * g\n"
        "        m_hat = m / (1 - beta1 ** step)\n"
        "        v_hat = v / (1 - beta2 ** step)\n"
        "        theta = theta - lr * (m_hat / (v_hat.sqrt() + eps))\n"
        "    theta_l2 = theta\n"
        "\n"
        "    # --- AdamW path: plain Adam grad, decay applied to theta directly ---\n"
        "    theta = theta0.clone()\n"
        "    m = t.zeros_like(theta)\n"
        "    v = t.zeros_like(theta)\n"
        "    for step in (1, 2):\n"
        "        g = grads[step - 1]\n"
        "        m = beta1 * m + (1 - beta1) * g\n"
        "        v = beta2 * v + (1 - beta2) * g * g\n"
        "        m_hat = m / (1 - beta1 ** step)\n"
        "        v_hat = v / (1 - beta2 ** step)\n"
        "        theta = theta - lr * (m_hat / (v_hat.sqrt() + eps)) - lr * lmda * theta\n"
        "    theta_adamw = theta\n"
        "\n"
        "    return theta_l2, theta_adamw"
    ),
    "solution_notes": (
        "**Two-step state matters.** A single Adam step with `m=v=0` is "
        "atypical — the bias correction `1 − β^1` exactly cancels the "
        "`1 − β` numerator coefficients, making `m̂/√v̂ = sign(g)`. Step 2 is "
        "where `m`, `v` actually carry forward and the L2-vs-AdamW gap "
        "manifests through the asymmetric `√v̂` normalisation.\n\n"
        "**Why bigger `lmda` ⇒ bigger gap.** Both paths apply MORE decay as "
        "`lmda` grows, but the L2 path routes that decay through Adam's "
        "`/√v̂`, which scales differently per coordinate. Larger `lmda` "
        "means larger contribution of `lmda·θ` inside `g`, larger relative "
        "asymmetry under normalisation.\n\n"
        "**`theta0.clone()` is the no-mutation contract.** Re-using the "
        "same starting tensor across both trajectories — without cloning — "
        "would link them. The caller passes in `theta0` once; we own the "
        "two trajectory copies."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 4 — dataloader-pin-memory-workers ex2 (seeded shuffle, batch shape)
# ---------------------------------------------------------------------------

SPEC_DL_WORKER_SEED = {
    "atom_id": "dataloader-pin-memory-workers",
    "subtopic": "PyTorch: DataLoader pin_memory + workers",
    "topic_folder": TOPIC_PT,
    "atom_recap_md": RECAP_DL_WORKER_SEED,
    "exercise_index": 2,
    "exercise_title": "DataLoader with shuffle=True + seeded worker_init_fn — verify batch shapes and full-epoch coverage",
    "slug": "dataloader-shuffle-worker-init-fn-batch-shape-invariant",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["dataloader", "worker_init_fn", "shuffle", "batch-shape"],
    "kcs": [
        "worker-init-fn-seeded-rng",
        "shuffle-preserves-batch-shape",
    ],
    "lo": (
        "Apply `worker_init_fn` + `torch.initial_seed()` to seed each "
        "worker's numpy/random RNG deterministically, and verify that under "
        "`shuffle=True` the resulting DataLoader still yields "
        "`(batch_size, *item_shape)` batches and covers every dataset item "
        "in one epoch."
    ),
    "prompt_body": (
        "Implement `ex2_make_seeded_shuffle_dataloader(dataset, batch_size, "
        "num_workers)`. Returns a `DataLoader` configured for:\n\n"
        "1. `batch_size` items per batch.\n"
        "2. `num_workers` worker processes (may be 0 for in-main).\n"
        "3. `shuffle=True`.\n"
        "4. `pin_memory=False` (CPU-only test environment).\n"
        "5. `drop_last=False` — keep the last partial batch.\n"
        "6. `worker_init_fn` that seeds numpy AND Python `random` per "
        "worker via `torch.initial_seed()`:\n"
        "```python\n"
        "def worker_init_fn(worker_id):\n"
        "    import numpy as np, random\n"
        "    base = torch.initial_seed() % (2**32)\n"
        "    np.random.seed(base + worker_id)\n"
        "    random.seed(base + worker_id)\n"
        "```\n\n"
        "Return the configured DataLoader.\n\n"
        "**The test then verifies:**\n"
        "- batch shape `(B, *item_shape)`, including last batch when "
        "`len(dataset) % batch_size != 0`.\n"
        "- total items across one epoch == `len(dataset)`.\n"
        "- two epochs over the same loader yield DIFFERENT orderings "
        "(shuffle works) but the same total item count.\n\n"
        "**Use `from torch.utils.data import DataLoader, TensorDataset`.**"
    ),
    "stub": (
        "from torch.utils.data import DataLoader, TensorDataset\n"
        "\n"
        "def ex2_make_seeded_shuffle_dataloader(dataset, batch_size: int, num_workers: int) -> DataLoader:\n"
        '    """Build a DataLoader with shuffle + seeded worker_init_fn (CPU-only)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "from torch.utils.data import TensorDataset, DataLoader\n"
        "\n"
        "# === Construction ===\n"
        "t.manual_seed(0)\n"
        "data = t.randn(11, 3, 4)   # 11 items, shape (3, 4); 11 is intentionally non-divisible\n"
        "labels = t.arange(11)\n"
        "ds = TensorDataset(data, labels)\n"
        "loader = ex2_make_seeded_shuffle_dataloader(ds, batch_size=4, num_workers=0)\n"
        "assert isinstance(loader, DataLoader), f'must return a DataLoader, got {type(loader).__name__}'\n"
        "\n"
        "# === Config readback ===\n"
        "assert loader.batch_size == 4\n"
        "assert loader.num_workers == 0\n"
        "assert loader.pin_memory is False, 'pin_memory must be False on CPU'\n"
        "assert loader.drop_last is False\n"
        "# Sampler must be a RandomSampler (shuffle=True).\n"
        "from torch.utils.data import RandomSampler\n"
        "assert isinstance(loader.sampler, RandomSampler), f'shuffle=True needs RandomSampler, got {type(loader.sampler).__name__}'\n"
        "# worker_init_fn must be set and callable.\n"
        "assert callable(loader.worker_init_fn), 'worker_init_fn must be set'\n"
        "\n"
        "# === Batch shape invariant under shuffle ===\n"
        "all_items = []\n"
        "batch_shapes = []\n"
        "for x_batch, y_batch in loader:\n"
        "    batch_shapes.append(tuple(x_batch.shape))\n"
        "    # Each non-last batch: (4, 3, 4). Last: (3, 3, 4).\n"
        "    assert x_batch.shape[1:] == (3, 4), f'item dims must be (3, 4), got {x_batch.shape[1:]}'\n"
        "    assert x_batch.shape[0] in (4, 3), f'batch size must be 4 or 3 (last partial), got {x_batch.shape[0]}'\n"
        "    all_items.append(y_batch)\n"
        "\n"
        "# === Total items covered ===\n"
        "flat = t.cat(all_items)\n"
        "assert len(flat) == 11, f'one epoch must yield all 11 items, got {len(flat)}'\n"
        "assert set(flat.tolist()) == set(range(11)), 'every dataset index must appear exactly once'\n"
        "\n"
        "# === Last batch is partial: 11 = 4 + 4 + 3 ===\n"
        "assert sorted([s[0] for s in batch_shapes]) == [3, 4, 4], f'expected batches of sizes [3,4,4], got {batch_shapes}'\n"
        "\n"
        "# === Two epochs: shuffle changes order, but total count stays 11 ===\n"
        "t.manual_seed(42)\n"
        "order_a = t.cat([y for _, y in loader]).tolist()\n"
        "t.manual_seed(43)\n"
        "order_b = t.cat([y for _, y in loader]).tolist()\n"
        "assert sorted(order_a) == sorted(order_b) == list(range(11)), 'both epochs cover full dataset'\n"
        "assert order_a != order_b, f'two epochs with different seeds should differ; got identical {order_a}'\n"
        "\n"
        "# === worker_init_fn actually seeds numpy ===\n"
        "# Direct call to verify the function reseeds numpy deterministically.\n"
        "wif = loader.worker_init_fn\n"
        "# Set torch initial_seed via manual_seed (which sets initial_seed in the current thread).\n"
        "t.manual_seed(12345)\n"
        "wif(0)\n"
        "a0 = np.random.rand()\n"
        "t.manual_seed(12345)\n"
        "wif(0)\n"
        "a1 = np.random.rand()\n"
        "assert a0 == a1, f'same torch seed + same worker_id must produce same numpy stream; got {a0} vs {a1}'\n"
        "t.manual_seed(12345)\n"
        "wif(1)\n"
        "a2 = np.random.rand()\n"
        "assert a0 != a2, f'different worker_id should produce different numpy stream; got identical {a0}'\n"
        "\n"
        "# === A divisible batch_size (8 % 4 == 0) → no partial batch ===\n"
        "ds_div = TensorDataset(t.randn(8, 2), t.arange(8))\n"
        "loader_div = ex2_make_seeded_shuffle_dataloader(ds_div, batch_size=4, num_workers=0)\n"
        "shapes = [b[0].shape[0] for b in loader_div]\n"
        "assert shapes == [4, 4], f'divisible: expected [4, 4], got {shapes}'"
    ),
    "solution_body": (
        "from torch.utils.data import DataLoader\n"
        "import torch as _t\n"
        "import numpy as _np\n"
        "import random as _random\n"
        "\n"
        "def _worker_init_fn(worker_id):\n"
        "    base = _t.initial_seed() % (2 ** 32)\n"
        "    _np.random.seed(base + worker_id)\n"
        "    _random.seed(base + worker_id)\n"
        "\n"
        "def ex2_make_seeded_shuffle_dataloader(dataset, batch_size, num_workers):\n"
        "    return DataLoader(\n"
        "        dataset,\n"
        "        batch_size=batch_size,\n"
        "        num_workers=num_workers,\n"
        "        shuffle=True,\n"
        "        pin_memory=False,\n"
        "        drop_last=False,\n"
        "        worker_init_fn=_worker_init_fn,\n"
        "    )"
    ),
    "solution_notes": (
        "**`torch.initial_seed()` inside a worker is the per-worker base "
        "seed PyTorch already assigned.** Calling it from a worker_init_fn "
        "lets you derive a numpy/random seed that's distinct per worker "
        "AND deterministic given `torch.manual_seed(...)` in the main "
        "process.\n\n"
        "**`% (2**32)` is required.** numpy and Python's `random` both want "
        "a uint32 seed. `torch.initial_seed()` returns a 63-bit int; the "
        "modulo collapses it safely.\n\n"
        "**`pin_memory=False` on CPU is correct.** `pin_memory=True` would "
        "raise at first batch fetch without CUDA. The contract we're "
        "exercising is the rest of the DataLoader config — workers, "
        "shuffle, seeded init — all of which work fine CPU-only."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 5 — 1x1-conv-channel-reshape ex2 (rectangular spatial preserved)
# ---------------------------------------------------------------------------

SPEC_CONV_SPATIAL = {
    "atom_id": "1x1-conv-channel-reshape",
    "subtopic": "CNN: 1x1 conv channel-reshape",
    "topic_folder": TOPIC_RES,
    "atom_recap_md": RECAP_CONV_SPATIAL,
    "exercise_index": 2,
    "exercise_title": "1×1 conv on rectangular (H≠W) input — output has same H, W and matches per-pixel Linear",
    "slug": "one-by-one-conv-rectangular-input-h-w-preserved",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["conv2d", "1x1", "rectangular", "per-pixel-linear"],
    "kcs": [
        "1x1-conv-preserves-spatial-axes",
        "conv-weight-reshape-to-linear-weight",
    ],
    "lo": (
        "Apply the per-pixel-Linear identity for a 1×1 Conv2d on a "
        "RECTANGULAR `(H ≠ W)` feature map: verify the output shape "
        "preserves both spatial axes AND matches an explicit "
        "`Linear(C_in, C_out)` computed pixel-by-pixel."
    ),
    "prompt_body": (
        "Implement `ex2_rect_one_by_one_via_linear(x, conv)`. Same idea as "
        "ex1, but the input has DISTINCT spatial dims `(H ≠ W)` — for "
        "instance `(B=2, C_in=3, H=7, W=11)`.\n\n"
        "Steps:\n"
        "1. Read `conv.weight` (shape `(C_out, C_in, 1, 1)`) and "
        "`conv.bias` (shape `(C_out,)` or `None`).\n"
        "2. Reshape the conv weight to a Linear weight of shape "
        "`(C_out, C_in)` — squeeze the trailing 1×1.\n"
        "3. Use `einops.rearrange` to fold `H, W` into a single batch-like "
        "axis: `'b c h w -> (b h w) c'`.\n"
        "4. Apply the Linear: `out_flat = x_flat @ W.T + bias` (broadcast "
        "bias correctly; if `conv.bias is None`, skip it).\n"
        "5. `rearrange` back to `(b, c_out, h, w)`.\n\n"
        "Inputs:\n"
        "- `x`: shape `(B, C_in, H, W)` with possibly `H ≠ W`.\n"
        "- `conv`: a `nn.Conv2d(C_in, C_out, kernel_size=1)`.\n\n"
        "Output: a `Tensor` of shape `(B, C_out, H, W)` — identical to "
        "`conv(x)` to within fp32 atol."
    ),
    "stub": (
        "def ex2_rect_one_by_one_via_linear(x: Tensor, conv) -> Tensor:\n"
        '    """Per-pixel Linear equivalent of 1×1 Conv2d, on a rectangular (H≠W) input."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "\n"
        "# === Rectangular input, with bias ===\n"
        "t.manual_seed(0)\n"
        "conv = nn.Conv2d(3, 5, kernel_size=1, bias=True)\n"
        "x = t.randn(2, 3, 7, 11)   # H=7, W=11 — intentionally rectangular\n"
        "out = ex2_rect_one_by_one_via_linear(x, conv)\n"
        "ref = conv(x)\n"
        "assert out.shape == (2, 5, 7, 11), f'output shape must be (2,5,7,11); got {tuple(out.shape)}'\n"
        "assert t.allclose(out, ref, atol=1e-5), f'output must match conv(x); maxabs={(out-ref).abs().max().item():.2e}'\n"
        "\n"
        "# === Spatial axes survive verbatim ===\n"
        "# After 1×1 conv, H and W must be UNCHANGED — same numbers, in the same slots.\n"
        "assert out.shape[2] == x.shape[2], f'H must be preserved: in {x.shape[2]} vs out {out.shape[2]}'\n"
        "assert out.shape[3] == x.shape[3], f'W must be preserved: in {x.shape[3]} vs out {out.shape[3]}'\n"
        "\n"
        "# === H and W truly distinct — no transpose bug ===\n"
        "# If a misimplementation swapped H↔W, on a square input you'd never notice.\n"
        "# Rectangular catches it. We explicitly verify (out, x) match dimwise.\n"
        "assert out.shape[2] != out.shape[3], 'this test must use H ≠ W (currently {out.shape[2]} vs {out.shape[3]})'\n"
        "\n"
        "# === Bias=None ===\n"
        "t.manual_seed(1)\n"
        "conv_nb = nn.Conv2d(4, 6, kernel_size=1, bias=False)\n"
        "x_nb = t.randn(1, 4, 5, 13)\n"
        "out_nb = ex2_rect_one_by_one_via_linear(x_nb, conv_nb)\n"
        "ref_nb = conv_nb(x_nb)\n"
        "assert out_nb.shape == (1, 6, 5, 13)\n"
        "assert t.allclose(out_nb, ref_nb, atol=1e-5)\n"
        "\n"
        "# === Extreme rectangular (H=1, W=20) ===\n"
        "conv_ex = nn.Conv2d(2, 3, kernel_size=1)\n"
        "x_ex = t.randn(2, 2, 1, 20)\n"
        "out_ex = ex2_rect_one_by_one_via_linear(x_ex, conv_ex)\n"
        "ref_ex = conv_ex(x_ex)\n"
        "assert out_ex.shape == (2, 3, 1, 20)\n"
        "assert t.allclose(out_ex, ref_ex, atol=1e-5)\n"
        "\n"
        "# === Extreme rectangular (H=17, W=1) — transpose to the OTHER extreme ===\n"
        "x_ex2 = t.randn(2, 2, 17, 1)\n"
        "out_ex2 = ex2_rect_one_by_one_via_linear(x_ex2, conv_ex)\n"
        "ref_ex2 = conv_ex(x_ex2)\n"
        "assert out_ex2.shape == (2, 3, 17, 1)\n"
        "assert t.allclose(out_ex2, ref_ex2, atol=1e-5)\n"
        "\n"
        "# === Output is a Tensor (not a numpy array etc) ===\n"
        "assert isinstance(out, t.Tensor)"
    ),
    "solution_body": (
        "import einops as _einops\n"
        "\n"
        "def ex2_rect_one_by_one_via_linear(x, conv):\n"
        "    W = conv.weight                                  # (C_out, C_in, 1, 1)\n"
        "    W_lin = W.squeeze(-1).squeeze(-1)                # (C_out, C_in)\n"
        "    b, c_in, h, w = x.shape\n"
        "    x_flat = _einops.rearrange(x, 'b c h w -> (b h w) c')  # ((b·h·w), C_in)\n"
        "    out_flat = x_flat @ W_lin.t()                    # ((b·h·w), C_out)\n"
        "    if conv.bias is not None:\n"
        "        out_flat = out_flat + conv.bias\n"
        "    return _einops.rearrange(\n"
        "        out_flat, '(b h w) c -> b c h w', b=b, h=h, w=w\n"
        "    )"
    ),
    "solution_notes": (
        "**`squeeze(-1).squeeze(-1)` over `.reshape(C_out, C_in)`.** Either "
        "works for a 1×1 conv. `squeeze` makes the no-spatial intent "
        "explicit; `reshape` would silently work for any kernel size, "
        "potentially masking a bug if the conv weren't actually 1×1.\n\n"
        "**`b=b, h=h, w=w` in the un-rearrange is load-bearing.** Without "
        "passing the axis sizes back, einops can't uniquely decompose the "
        "flat axis when `b·h·w` factors ambiguously (e.g. 2·7·11 = 154, but "
        "also 11·14 or 7·22). Explicit sizes pin the correct factorisation.\n\n"
        "**Why rectangular catches the transpose bug.** A naive "
        "implementation that reshapes via `'b c h w -> (b w h) c'` (note: w "
        "before h) and reverses with `'(b h w) c'` will silently swap H ↔ W. "
        "On a square input the swap is invisible — the test would pass. "
        "Rectangular inputs make this bug an immediate shape mismatch."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 6 — batchnorm-running-stats ex2 (.train() vs .eval())
# ---------------------------------------------------------------------------

SPEC_BN_EVAL = {
    "atom_id": "batchnorm-running-stats",
    "subtopic": "CNN: BatchNorm running stats",
    "topic_folder": TOPIC_RES,
    "atom_recap_md": RECAP_BN_EVAL,
    "exercise_index": 2,
    "exercise_title": "train/eval BN forward: train uses batch stats; eval uses running stats — and outputs differ",
    "slug": "batchnorm-train-vs-eval-uses-different-stats",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["batchnorm", "train", "eval", "running-stats"],
    "kcs": [
        "train-mode-uses-batch-mean-var",
        "eval-mode-uses-running-mean-var",
    ],
    "lo": (
        "Analyse the BatchNorm forward pass under `.train()` vs `.eval()`: "
        "train mode normalises by the BATCH's mean/var (and updates the "
        "running stats), while eval mode normalises by the FIXED running "
        "stats — so the same `x` through the same `bn` produces DIFFERENT "
        "outputs in the two modes."
    ),
    "prompt_body": (
        "Implement `ex2_bn_train_vs_eval_forward(x, bn)`. Same `bn` "
        "module, same input `x`, run forward in both modes.\n\n"
        "Steps:\n"
        "1. `bn.train()` → call `y_train = bn(x)`. This uses the batch's "
        "OWN mean/var AND updates `bn.running_mean` / `bn.running_var` "
        "via the EMA (PyTorch handles this internally).\n"
        "2. `bn.eval()` → call `y_eval = bn(x)`. This uses the "
        "(now-updated) `running_mean` / `running_var` and does NOT "
        "update them.\n"
        "3. Return tuple `(y_train, y_eval)`.\n\n"
        "Do NOT manually rebuild the BN formula — call `bn(x)` and let "
        "PyTorch swap the stats based on the mode. The drill is about "
        "OBSERVING the mode-dependent behaviour, not re-deriving it.\n\n"
        "Inputs:\n"
        "- `x`: `(B, C, H, W)` for `BatchNorm2d`, or `(B, C)` for "
        "`BatchNorm1d`.\n"
        "- `bn`: an `nn.BatchNorm2d` or `nn.BatchNorm1d` instance with "
        "pre-existing `running_mean` / `running_var` (defaults are 0 / 1).\n\n"
        "Output: `(y_train, y_eval)` — both Tensors of shape `x.shape`."
    ),
    "stub": (
        "def ex2_bn_train_vs_eval_forward(x: Tensor, bn) -> tuple:\n"
        '    """Run bn(x) in train() and eval() modes; return (y_train, y_eval)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "\n"
        "# === Train vs eval outputs DIFFER (the whole point) ===\n"
        "t.manual_seed(0)\n"
        "bn = nn.BatchNorm2d(4)\n"
        "x = t.randn(8, 4, 5, 6) * 3.0 + 2.0   # non-unit stats so the two modes disagree\n"
        "y_train, y_eval = ex2_bn_train_vs_eval_forward(x, bn)\n"
        "assert y_train.shape == x.shape and y_eval.shape == x.shape\n"
        "diff = (y_train - y_eval).abs()\n"
        "assert diff.max() > 1e-3, (\n"
        "    f'train and eval outputs must differ on non-trivial input; got max diff {diff.max().item():.4e}'\n"
        ")\n"
        "\n"
        "# === Train mode: output's per-channel batch mean ≈ 0 (after centering) ===\n"
        "# In train mode, the batch is centered by its own mean; per-channel\n"
        "# spatial+batch mean of the output should be ≈ bn.bias (init=0) and var\n"
        "# scaled by bn.weight (init=1).\n"
        "ch_mean = y_train.mean(dim=(0, 2, 3))   # (C,)\n"
        "assert t.allclose(ch_mean, t.zeros(4), atol=1e-4), (\n"
        "    f'train-mode per-channel batch mean must be ≈0, got {ch_mean}'\n"
        ")\n"
        "ch_var = y_train.var(dim=(0, 2, 3), unbiased=False)\n"
        "assert t.allclose(ch_var, t.ones(4), atol=1e-3), (\n"
        "    f'train-mode per-channel batch var must be ≈1, got {ch_var}'\n"
        ")\n"
        "\n"
        "# === Eval mode: running stats actually used ===\n"
        "# After the train pass, running_mean ≠ 0 and running_var ≠ 1.\n"
        "rm = bn.running_mean.clone()\n"
        "rv = bn.running_var.clone()\n"
        "assert not t.allclose(rm, t.zeros(4)), f'running_mean must have moved off 0 after train pass, got {rm}'\n"
        "assert not t.allclose(rv, t.ones(4)), f'running_var must have moved off 1 after train pass, got {rv}'\n"
        "# y_eval should match a manual computation using the running stats.\n"
        "manual_eval = (x - rm[None, :, None, None]) / t.sqrt(rv[None, :, None, None] + bn.eps)\n"
        "manual_eval = manual_eval * bn.weight[None, :, None, None] + bn.bias[None, :, None, None]\n"
        "assert t.allclose(y_eval, manual_eval, atol=1e-5), (\n"
        "    f'eval output must use running stats; mismatch max {(y_eval - manual_eval).abs().max().item():.2e}'\n"
        ")\n"
        "\n"
        "# === Eval mode does NOT update running stats ===\n"
        "t.manual_seed(1)\n"
        "bn2 = nn.BatchNorm2d(3)\n"
        "x2 = t.randn(4, 3, 2, 2) * 5.0\n"
        "bn2.train()\n"
        "bn2(x2)                                   # one train pass to move running stats\n"
        "rm_before = bn2.running_mean.clone()\n"
        "rv_before = bn2.running_var.clone()\n"
        "bn2.eval()\n"
        "for _ in range(3):\n"
        "    bn2(x2)                               # multiple eval passes — must not change running stats\n"
        "assert t.allclose(bn2.running_mean, rm_before), 'eval mode must NOT update running_mean'\n"
        "assert t.allclose(bn2.running_var, rv_before), 'eval mode must NOT update running_var'\n"
        "\n"
        "# === Mode is left in eval() after the function returns ===\n"
        "# We don't strictly require this, but it's the natural last step.\n"
        "# Test instead that the function doesn't break with consecutive calls.\n"
        "y_t2, y_e2 = ex2_bn_train_vs_eval_forward(x2, bn2)\n"
        "assert y_t2.shape == x2.shape and y_e2.shape == x2.shape\n"
        "\n"
        "# === Return type is a 2-tuple of Tensors ===\n"
        "ret = ex2_bn_train_vs_eval_forward(x, bn)\n"
        "assert isinstance(ret, tuple) and len(ret) == 2\n"
        "assert isinstance(ret[0], t.Tensor) and isinstance(ret[1], t.Tensor)"
    ),
    "solution_body": (
        "def ex2_bn_train_vs_eval_forward(x, bn):\n"
        "    bn.train()\n"
        "    y_train = bn(x)\n"
        "    bn.eval()\n"
        "    y_eval = bn(x)\n"
        "    return y_train, y_eval"
    ),
    "solution_notes": (
        "**Mode toggling is the whole API.** PyTorch's BN does the "
        "stat-swap internally based on `self.training`. The Module-level "
        "`.train()` / `.eval()` flip flips this flag on every child "
        "(including BNs) — no need to thread a `training=True` arg through.\n\n"
        "**Train pass UPDATES running stats; eval pass does NOT.** The "
        "second test confirms this by capturing `running_mean`/`var` "
        "after a train pass, then running multiple eval passes and "
        "checking the stats haven't moved.\n\n"
        "**Why this is the canonical deploy bug.** A user who forgets to "
        "call `model.eval()` before inference gets the BN-in-train-mode "
        "behaviour: each inference batch's own stats are used. For a "
        "single-image inference call, batch_var ≈ 0 → division by ε → "
        "garbled outputs. The two-line `model.eval()` fix is invisible "
        "from the loss curves during training."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 7 — maxpool-reduce ex2 (max vs avg via einops.reduce)
# ---------------------------------------------------------------------------

SPEC_MAX_VS_AVG = {
    "atom_id": "maxpool-reduce",
    "subtopic": "CNN: MaxPool as reduce",
    "topic_folder": TOPIC_RES,
    "atom_recap_md": RECAP_MAX_VS_AVG,
    "exercise_index": 2,
    "exercise_title": "swap 'max' → 'mean' in einops.reduce to build AvgPool2d alongside MaxPool2d",
    "slug": "avg-pool-vs-max-pool-via-einops-reduce",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["maxpool", "avgpool", "einops", "reduce"],
    "kcs": [
        "einops-reduce-op-swap",
        "pool-output-shape-h-over-p-w-over-p",
    ],
    "lo": (
        "Apply the einops `reduce` pattern with TWO ops on the same window "
        "geometry — `'max'` to build `MaxPool2d` and `'mean'` to build "
        "`AvgPool2d` — and verify both outputs have shape "
        "`(..., H/p, W/p)` while diverging on a contrived input where "
        "max-pool preserves outliers that avg-pool dilutes."
    ),
    "prompt_body": (
        "Implement `ex2_max_vs_avg_pool(x, p)`. Returns the tuple "
        "`(max_out, avg_out)` computed via `einops.reduce` with `'max'` "
        "and `'mean'` over the same window pattern.\n\n"
        "Steps:\n"
        "1. `max_out = reduce(x, '... (h p1) (w p2) -> ... h w', 'max', "
        "p1=p, p2=p)`.\n"
        "2. `avg_out = reduce(x, '... (h p1) (w p2) -> ... h w', 'mean', "
        "p1=p, p2=p)`.\n"
        "3. Return `(max_out, avg_out)`.\n\n"
        "Inputs:\n"
        "- `x`: at least 2D. Trailing axes are `(..., H, W)` with `H % p == "
        "0` and `W % p == 0`.\n"
        "- `p`: window size (int).\n\n"
        "Output: tuple of Tensors. Both have shape `(..., H/p, W/p)`. The "
        "test will compare to `nn.MaxPool2d` / `nn.AvgPool2d` on 4D "
        "inputs."
    ),
    "stub": (
        "def ex2_max_vs_avg_pool(x: Tensor, p: int) -> tuple:\n"
        '    """MaxPool + AvgPool over (p, p) windows via einops.reduce."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "\n"
        "# === Shape invariant: both outputs have shape (..., H/p, W/p) ===\n"
        "x = t.randn(2, 3, 6, 8)\n"
        "p = 2\n"
        "max_out, avg_out = ex2_max_vs_avg_pool(x, p)\n"
        "assert max_out.shape == (2, 3, 3, 4), f'max shape wrong: {tuple(max_out.shape)}'\n"
        "assert avg_out.shape == (2, 3, 3, 4), f'avg shape wrong: {tuple(avg_out.shape)}'\n"
        "\n"
        "# === Max matches nn.MaxPool2d ===\n"
        "ref_max = nn.MaxPool2d(kernel_size=p)(x)\n"
        "assert t.allclose(max_out, ref_max, atol=1e-6), f'max != nn.MaxPool2d; max diff {(max_out-ref_max).abs().max().item():.2e}'\n"
        "\n"
        "# === Avg matches nn.AvgPool2d ===\n"
        "ref_avg = nn.AvgPool2d(kernel_size=p)(x)\n"
        "assert t.allclose(avg_out, ref_avg, atol=1e-6), f'avg != nn.AvgPool2d; max diff {(avg_out-ref_avg).abs().max().item():.2e}'\n"
        "\n"
        "# === Max != Avg in general — they MUST differ on real inputs ===\n"
        "diff = (max_out - avg_out).abs()\n"
        "assert diff.max() > 1e-3, f'max and avg pool must produce different outputs on random input; got max diff {diff.max().item():.4e}'\n"
        "\n"
        "# === The sparse-activation case: only one pixel per 2×2 is non-zero ===\n"
        "# Max preserves the bright pixel; avg dilutes it by 1/p² = 1/4.\n"
        "x_sparse = t.zeros(1, 1, 2, 2)\n"
        "x_sparse[0, 0, 0, 0] = 4.0\n"
        "m, a = ex2_max_vs_avg_pool(x_sparse, 2)\n"
        "assert m.item() == 4.0, f'max should preserve the bright pixel; got {m.item()}'\n"
        "assert a.item() == 1.0, f'avg should dilute 4.0 to 4/4=1.0; got {a.item()}'\n"
        "\n"
        "# === 3D input also works (no batch axis) — '...' covers it ===\n"
        "x_3d = t.randn(3, 4, 6)\n"
        "m3, a3 = ex2_max_vs_avg_pool(x_3d, 2)\n"
        "assert m3.shape == (3, 2, 3) and a3.shape == (3, 2, 3)\n"
        "\n"
        "# === p=1 → identity for both ===\n"
        "x_id = t.randn(2, 4, 4)\n"
        "m_id, a_id = ex2_max_vs_avg_pool(x_id, 1)\n"
        "assert t.allclose(m_id, x_id, atol=1e-6), 'p=1: max-pool is identity'\n"
        "assert t.allclose(a_id, x_id, atol=1e-6), 'p=1: avg-pool is identity'\n"
        "\n"
        "# === Window p=3 on a (3,3) image collapses to scalar per leading axis ===\n"
        "x_full = t.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])\n"
        "m_f, a_f = ex2_max_vs_avg_pool(x_full, 3)\n"
        "assert m_f.shape == (1, 1) and a_f.shape == (1, 1)\n"
        "assert m_f.item() == 9.0\n"
        "assert abs(a_f.item() - 5.0) < 1e-6   # mean of 1..9 is 5.0\n"
        "\n"
        "# === Return type ===\n"
        "ret = ex2_max_vs_avg_pool(x, p)\n"
        "assert isinstance(ret, tuple) and len(ret) == 2"
    ),
    "solution_body": (
        "import einops as _einops\n"
        "\n"
        "def ex2_max_vs_avg_pool(x, p):\n"
        "    max_out = _einops.reduce(\n"
        "        x, '... (h p1) (w p2) -> ... h w', 'max', p1=p, p2=p\n"
        "    )\n"
        "    avg_out = _einops.reduce(\n"
        "        x, '... (h p1) (w p2) -> ... h w', 'mean', p1=p, p2=p\n"
        "    )\n"
        "    return max_out, avg_out"
    ),
    "solution_notes": (
        "**Same pattern, different op string.** The einops `reduce` "
        "pattern `'... (h p1) (w p2) -> ... h w'` factors each spatial "
        "axis into `(h, p)` then reduces over `p`. The op string — "
        "`'max'`, `'mean'`, `'sum'`, `'min'`, `'prod'` — picks the "
        "aggregation. Same shape contract regardless.\n\n"
        "**Why `'...'` over an explicit `'b c'`.** The leading axes don't "
        "participate in the reduction; einops's `...` matches any number "
        "of them. This lets the same function handle 4D `(B, C, H, W)` "
        "and 3D `(C, H, W)` and even raw `(H, W)` images.\n\n"
        "**The sparse-activation contrast** is the load-bearing test. "
        "Random inputs give a quantitative `max ≠ avg` but the magnitudes "
        "depend on the seed. A one-hot 2×2 with a 4.0 spike isolates the "
        "qualitative difference: max=4.0 (identical to input), avg=1.0 "
        "(diluted by `1/p² = 1/4`)."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 8 — register-buffer ex2 (buffer vs Parameter contrast)
# ---------------------------------------------------------------------------

SPEC_BUF_VS_PARAM = {
    "atom_id": "register-buffer",
    "subtopic": "PyTorch: register_buffer",
    "topic_folder": TOPIC_RES,
    "atom_recap_md": RECAP_BUF_VS_PARAM,
    "exercise_index": 2,
    "exercise_title": "two-state Module: one buffer + one Parameter — show buffer is in state_dict but NOT in .parameters()",
    "slug": "buffer-vs-parameter-state-dict-parameters-membership",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["register_buffer", "nn.Parameter", "state_dict", "parameters"],
    "kcs": [
        "buffer-in-state-dict-not-in-parameters",
        "parameter-in-both-state-dict-and-parameters",
    ],
    "lo": (
        "Analyse the membership contract of `register_buffer` vs "
        "`nn.Parameter` by building a Module with ONE of each, then "
        "verifying: both appear in `state_dict()`; only the Parameter "
        "appears in `.parameters()`; neither requires_grad propagates to "
        "the buffer."
    ),
    "prompt_body": (
        "Implement `ex2_MixedStateModule` as a subclass of `nn.Module`:\n\n"
        "Constructor `__init__(self, dim)`:\n"
        "1. `super().__init__()`.\n"
        "2. Register a NON-TRAINABLE buffer named `'fixed_scale'`, "
        "initialised to `torch.ones(dim)` — use `self.register_buffer(...)`.\n"
        "3. Register a TRAINABLE parameter named `'weight'`, initialised to "
        "`torch.zeros(dim)` — use `nn.Parameter(...)` and assignment.\n\n"
        "Forward `forward(self, x)`:\n"
        "- Return `self.fixed_scale * x + self.weight`. (A simple affine "
        "for the test to exercise; nothing tricky.)\n\n"
        "The test will verify state_dict / parameters membership and "
        "behaviour under `.to(device)` and `requires_grad`.\n\n"
        "Inputs to `forward`: `x` of shape `(..., dim)`. Output: same "
        "shape."
    ),
    "stub": (
        "import torch.nn as nn\n"
        "\n"
        "class ex2_MixedStateModule(nn.Module):\n"
        "    def __init__(self, dim: int):\n"
        '        """Register one buffer + one Parameter; verify membership in state_dict / parameters."""\n'
        "        super().__init__()\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    def forward(self, x):\n"
        "        raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "\n"
        "# === Construction ===\n"
        "m = ex2_MixedStateModule(4)\n"
        "assert isinstance(m, nn.Module)\n"
        "\n"
        "# === Buffer present, initialised to ones ===\n"
        "assert hasattr(m, 'fixed_scale'), 'must register buffer fixed_scale'\n"
        "assert t.allclose(m.fixed_scale, t.ones(4)), f'fixed_scale init wrong: {m.fixed_scale}'\n"
        "assert not isinstance(m.fixed_scale, nn.Parameter), 'fixed_scale must NOT be an nn.Parameter'\n"
        "\n"
        "# === Parameter present, initialised to zeros ===\n"
        "assert hasattr(m, 'weight'), 'must register parameter weight'\n"
        "assert isinstance(m.weight, nn.Parameter), 'weight must be nn.Parameter'\n"
        "assert t.allclose(m.weight.detach(), t.zeros(4)), f'weight init wrong: {m.weight}'\n"
        "\n"
        "# === state_dict contains BOTH ===\n"
        "sd = m.state_dict()\n"
        "assert 'fixed_scale' in sd, 'buffer must be in state_dict'\n"
        "assert 'weight' in sd, 'parameter must be in state_dict'\n"
        "assert len(sd) == 2, f'state_dict should have exactly 2 entries, got {len(sd)}: {list(sd.keys())}'\n"
        "\n"
        "# === .parameters() contains ONLY the Parameter ===\n"
        "params = list(m.parameters())\n"
        "assert len(params) == 1, f'.parameters() must contain exactly 1 entry (weight), got {len(params)}'\n"
        "# It's `weight`:\n"
        "assert params[0].data_ptr() == m.weight.data_ptr(), '.parameters() entry must be weight'\n"
        "\n"
        "# === named_buffers contains ONLY fixed_scale ===\n"
        "bufs = dict(m.named_buffers())\n"
        "assert 'fixed_scale' in bufs and len(bufs) == 1, f'named_buffers wrong: {list(bufs.keys())}'\n"
        "\n"
        "# === named_parameters contains ONLY weight ===\n"
        "nps = dict(m.named_parameters())\n"
        "assert 'weight' in nps and len(nps) == 1, f'named_parameters wrong: {list(nps.keys())}'\n"
        "\n"
        "# === requires_grad: weight yes, buffer no ===\n"
        "assert m.weight.requires_grad is True, 'weight must require_grad'\n"
        "assert m.fixed_scale.requires_grad is False, 'fixed_scale must NOT require_grad (buffers do not)'\n"
        "\n"
        "# === Forward works and uses both ===\n"
        "x = t.randn(3, 4)\n"
        "y = m(x)\n"
        "assert y.shape == x.shape\n"
        "# With weight=0 and fixed_scale=1, forward should equal x.\n"
        "assert t.allclose(y, x, atol=1e-6), f'init forward should equal x (scale=1, bias=0), got {y}'\n"
        "\n"
        "# === Backward flows ONLY to weight, never to buffer ===\n"
        "loss = m(x).sum()\n"
        "loss.backward()\n"
        "assert m.weight.grad is not None, 'weight must receive gradient'\n"
        "assert not t.allclose(m.weight.grad, t.zeros(4)), 'weight grad should be non-zero'\n"
        "# Buffer has no grad attribute populated (requires_grad=False).\n"
        "assert m.fixed_scale.grad is None, f'buffer must NOT accumulate grad, got {m.fixed_scale.grad}'\n"
        "\n"
        "# === optimizer.step() updates weight, leaves buffer untouched ===\n"
        "m2 = ex2_MixedStateModule(4)\n"
        "opt = t.optim.SGD(m2.parameters(), lr=0.1)\n"
        "scale_before = m2.fixed_scale.clone()\n"
        "weight_before = m2.weight.detach().clone()\n"
        "out = m2(t.ones(2, 4))\n"
        "out.sum().backward()\n"
        "opt.step()\n"
        "assert t.allclose(m2.fixed_scale, scale_before), 'optimizer.step() must NOT touch the buffer'\n"
        "assert not t.allclose(m2.weight.detach(), weight_before), 'optimizer.step() must update weight'\n"
        "\n"
        "# === state_dict round-trip preserves the buffer ===\n"
        "m3 = ex2_MixedStateModule(4)\n"
        "with t.no_grad():\n"
        "    m3.fixed_scale.copy_(t.tensor([1.5, 2.0, 0.5, 0.25]))\n"
        "    m3.weight.copy_(t.tensor([0.1, 0.2, 0.3, 0.4]))\n"
        "m4 = ex2_MixedStateModule(4)\n"
        "m4.load_state_dict(m3.state_dict())\n"
        "assert t.allclose(m4.fixed_scale, m3.fixed_scale), 'state_dict must save+load the buffer'\n"
        "assert t.allclose(m4.weight.detach(), m3.weight.detach()), 'state_dict must save+load the parameter'"
    ),
    "solution_body": (
        "import torch as _t\n"
        "import torch.nn as nn\n"
        "\n"
        "class ex2_MixedStateModule(nn.Module):\n"
        "    def __init__(self, dim):\n"
        "        super().__init__()\n"
        "        self.register_buffer('fixed_scale', _t.ones(dim))\n"
        "        self.weight = nn.Parameter(_t.zeros(dim))\n"
        "\n"
        "    def forward(self, x):\n"
        "        return self.fixed_scale * x + self.weight"
    ),
    "solution_notes": (
        "**`register_buffer(name, tensor)` over `self.name = tensor`.** "
        "A plain attribute assignment puts the tensor on the module but "
        "OUT of state_dict — it won't save/load, won't move with "
        "`.to(device)`. `register_buffer` is the documented hook.\n\n"
        "**`nn.Parameter(tensor)` automatically registers in both "
        "state_dict and .parameters().** No `register_parameter` call "
        "needed when you assign via `self.weight = nn.Parameter(...)` — "
        "`nn.Module.__setattr__` notices the type and wires it up.\n\n"
        "**The optimizer-step test is the load-bearing one.** It's not "
        "just about API surface (`'in state_dict'` vs `'in parameters'`); "
        "it's about RUNTIME effect: passing `model.parameters()` to "
        "`SGD(...)` automatically excludes buffers from updates. This is "
        "why BN's running_mean stays fixed under SGD even though it "
        "appears in `state_dict`."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# All specs
# ---------------------------------------------------------------------------

SPECS = [
    SPEC_NESTEROV,
    SPEC_ADAMW_VS_COUPLED,
    SPEC_L2_INTO_ADAM,
    SPEC_DL_WORKER_SEED,
    SPEC_CONV_SPATIAL,
    SPEC_BN_EVAL,
    SPEC_MAX_VS_AVG,
    SPEC_BUF_VS_PARAM,
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
    print(f"[deepening_y_batch12] Verifying {len(SPECS)} specs...")
    _verify_all(SPECS)

    print(f"\n[deepening_y_batch12] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_y_batch12] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
