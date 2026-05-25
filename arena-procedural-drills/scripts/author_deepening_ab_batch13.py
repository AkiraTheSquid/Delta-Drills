#!/usr/bin/env python3
"""Author 6 ex2 deepening drills (batch 13, group Ab — VAE + GAN cluster).

Atoms (all topic prereqs_vae_gan):
    - discriminator-classifier-head       (ex2: PatchGAN-style 1x1-conv per-patch head)
    - elbo-loss-sum-with-beta             (ex2: contrast beta=0 vs 1 vs 4 on per-sample loss)
    - generator-project-and-reshape       (ex2: project to 8x8 then 2 upsamples to 32x32, verify shapes)
    - kl-divergence-gaussian-closed-form  (ex2: scalar 1-d vs per-dim sum + batch-mean, numeric check)
    - mu-logsigma-encoder-head            (ex2: two-head variant (separate Linears) matches single-head+chunk)
    - reparameterization-trick            (ex2: gradient flow check — grad on mu AND logsigma)

Each ex2 = DEEPENING facet from ex1. ONE LO + ONE Bloom + <=2 KCs per drill.

Note: all drills use `nn.Module` etc.; standard preamble has only `import torch as t`,
so we inject `import torch.nn as nn` + `import torch.nn.functional as F` via extra_imports.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_vae_gan"

EXTRA_NN_IMPORTS = [
    "import torch.nn as nn",
    "import torch.nn.functional as F",
]


# ---------------------------------------------------------------------------
# Recap blocks
# ---------------------------------------------------------------------------

RECAP_PATCHGAN = (
    "## PatchGAN-style classifier head — 1x1 conv → per-patch sigmoid map\n"
    "\n"
    "Ex1 used `Flatten → Linear(D, 1) → Sigmoid` to produce ONE scalar real/fake\n"
    "score per image. The deepening move is the **PatchGAN** head: instead of\n"
    "collapsing to a single number, keep the spatial map and produce ONE score\n"
    "PER PATCH using a `1x1 Conv2d(C, 1)` followed by sigmoid.\n"
    "\n"
    "```python\n"
    "self.head = nn.Conv2d(C_in, 1, kernel_size=1)   # 1x1 conv: per-pixel linear\n"
    "# forward:\n"
    "logits = self.head(features)         # (B, 1, H, W) — one logit per patch\n"
    "probs  = t.sigmoid(logits)           # (B, 1, H, W) — per-patch real/fake\n"
    "```\n"
    "\n"
    "**Why 1x1 conv = per-position Linear.** A 1x1 conv with `C_in→1` is\n"
    "exactly a Linear(C_in, 1) applied independently at every (h, w) cell of\n"
    "the feature map. Shared weights across positions, identical to weight-tying\n"
    "the linear head.\n"
    "\n"
    "**Loss with PatchGAN.** Downstream BCE loss takes the per-patch sigmoid\n"
    "map against a target map of all-ones (real) or all-zeros (fake) of\n"
    "matching shape. The discriminator is forced to make decisions per local\n"
    "receptive field rather than for the whole image — Pix2Pix's trick for\n"
    "high-frequency texture realism."
)

RECAP_BETA_CONTRAST = (
    "## ELBO with beta — three regimes on the same batch\n"
    "\n"
    "Ex1 computed `loss = recon + beta * kl`. The deepening move is to FEEL\n"
    "what `beta` does by contrasting three regimes on identical data:\n"
    "\n"
    "| beta | name        | behaviour                                       |\n"
    "| ---- | ----------- | ----------------------------------------------- |\n"
    "| 0    | pure recon  | autoencoder — latent unconstrained, perfect copy |\n"
    "| 1    | standard ELBO | balance: latent regularized to N(0, I)         |\n"
    "| 4    | β-VAE       | KL dominates — latent collapses, recon worsens  |\n"
    "\n"
    "```python\n"
    "recon_per_sample = ((x_hat - x) ** 2).flatten(1).sum(dim=1)    # (B,)\n"
    "kl_per_sample    = -0.5 * (1 + 2*logsigma - mu**2 - (2*logsigma).exp()).sum(dim=1)  # (B,)\n"
    "loss_per_sample  = recon_per_sample + beta * kl_per_sample     # (B,)\n"
    "```\n"
    "\n"
    "**Why per-sample, not scalar.** Returning the `(B,)` vector lets you\n"
    "inspect WHICH samples the model is bad at, AND lets downstream code\n"
    "decide whether to `.mean()` or `.sum()` for the gradient step.\n"
    "\n"
    "**Invariant:** at fixed recon + KL, `loss(beta=4) >= loss(beta=1) >=\n"
    "loss(beta=0)` for every sample (KL is non-negative). That ordering IS\n"
    "the test."
)

RECAP_8X8_PROJECT = (
    "## Project + reshape to 8x8 seed, then 2 upsamples to 32x32\n"
    "\n"
    "Ex1 projected latent `z` → 4x4 spatial seed. The deepening move is a\n"
    "DIFFERENT seed size (8x8) followed by two `ConvTranspose2d` upsamples\n"
    "to land at 32x32. The intermediate shapes are the load-bearing thing.\n"
    "\n"
    "```\n"
    "z: (B, latent_dim)\n"
    "  -> Linear(latent_dim, C*8*8) -> view (B, C, 8, 8)        # spatial seed\n"
    "  -> ConvTranspose2d(C, C//2, k=4, s=2, p=1) -> (B, C//2, 16, 16)\n"
    "  -> ConvTranspose2d(C//2, out_C, k=4, s=2, p=1) -> (B, out_C, 32, 32)\n"
    "```\n"
    "\n"
    "**Why 8x8 not 4x4.** A larger seed gives more spatial info to the\n"
    "early layers — fewer aggressive 2x upsamples needed. Pix2Pix and many\n"
    "DCGAN variants pick the seed to match `final_H >> n_upsamples` — at\n"
    "32x32 with 2 upsamples that's `32 >> 2 = 8`.\n"
    "\n"
    "**Seed reshape pattern.** `Linear(latent_dim, C*H*W).view(B, C, H, W)`.\n"
    "The Linear is doing all three jobs: dimensionality blow-up, learned\n"
    "spatial layout, AND the only non-conv mixing in the generator."
)

RECAP_KL_TWO_FORMS = (
    "## Gaussian KL — scalar 1-d closed form vs per-dim sum + batch mean\n"
    "\n"
    "Ex1 computed the per-sample KL for a multi-dim diagonal Gaussian. The\n"
    "deepening move shows the SAME formula in two equivalent forms:\n"
    "\n"
    "**Form A — scalar 1-d closed form, for `q = N(mu, sigma^2)` against `N(0,1)`:**\n"
    "```\n"
    "kl_scalar = 0.5 * (mu^2 + sigma^2 - 1 - 2 * log(sigma))\n"
    "          = 0.5 * (mu^2 + exp(2*logsigma) - 1 - 2 * logsigma)\n"
    "```\n"
    "\n"
    "**Form B — per-dim sum over a diagonal Gaussian, then mean over batch:**\n"
    "```\n"
    "kl_per_sample = -0.5 * sum_d(1 + 2*logsigma_d - mu_d^2 - exp(2*logsigma_d))\n"
    "kl_batch_mean = kl_per_sample.mean()\n"
    "```\n"
    "\n"
    "**They agree numerically.** Form B applied to a single dim (sum over\n"
    "one element) gives the same number as Form A for that dim. Run the\n"
    "scalar form on each dim independently, sum them — must equal Form B's\n"
    "per-sample KL.\n"
    "\n"
    "**Sign sanity.** At `mu=0, logsigma=0` (which means sigma=1), the KL\n"
    "is exactly 0 — `q == N(0, 1)` so no divergence. Any other (mu, logsigma)\n"
    "gives a strictly positive value. Useful unit test."
)

RECAP_TWO_HEAD = (
    "## Encoder head — two-Linear variant equivalent to single-Linear+chunk\n"
    "\n"
    "Ex1 used `Linear(d_in, 2*latent_dim)` then `.chunk(2, dim=-1)` to split\n"
    "into `(mu, logsigma)`. The deepening move is the TWO-HEAD form:\n"
    "\n"
    "```python\n"
    "class TwoHeadEncoder(nn.Module):\n"
    "    def __init__(self, d_in, latent_dim):\n"
    "        super().__init__()\n"
    "        self.fc_mu       = nn.Linear(d_in, latent_dim)\n"
    "        self.fc_logsigma = nn.Linear(d_in, latent_dim)\n"
    "    def forward(self, h):\n"
    "        return self.fc_mu(h), self.fc_logsigma(h)\n"
    "```\n"
    "\n"
    "**Both forms are equivalent** when initialized correctly. Functionally\n"
    "you can copy the single-Linear's weight matrix into the two heads:\n"
    "row-block `[0:latent_dim]` → `fc_mu`, row-block `[latent_dim:]` →\n"
    "`fc_logsigma`. Same forward.\n"
    "\n"
    "**Why two heads in practice.** Lets you put `nn.Softplus` or\n"
    "`Tanh()` only on the `logsigma` branch without touching `mu`.\n"
    "Conditional/asymmetric initialization (e.g. small `fc_logsigma`\n"
    "weight init for stable early KL) becomes trivial. Single-Linear forces\n"
    "shared init on both halves.\n"
    "\n"
    "**Parameter count is identical.** Two `Linear(d_in, k)` =\n"
    "`Linear(d_in, 2k)` parameter-wise. No FLOPs difference."
)

RECAP_GRADIENT_CHECK = (
    "## Reparameterization trick — gradient flow check\n"
    "\n"
    "Ex1 implemented `z = mu + sigma * eps`. The deepening move asks: WHY\n"
    "did we go through that algebra at all? Because the gradient needs to\n"
    "flow back to BOTH `mu` and `logsigma`.\n"
    "\n"
    "```python\n"
    "# WITH reparam — gradient flows:\n"
    "z = mu + (0.5 * logsigma).exp() * eps   # eps = N(0,1).sample()\n"
    "z.sum().backward()                       # mu.grad and logsigma.grad both populated\n"
    "\n"
    "# WITHOUT reparam — gradient DOES NOT flow:\n"
    "z = t.distributions.Normal(mu, sigma).sample()   # .sample() is non-differentiable\n"
    "z.sum().backward()                                # RuntimeError or zero grads\n"
    "```\n"
    "\n"
    "**Sampling is the source of randomness; reparam moves it OUT of the\n"
    "computational graph.** `eps ~ N(0, 1)` is sampled with `.detach()`-like\n"
    "semantics. The DETERMINISTIC transform `mu + exp(0.5*logsigma) * eps`\n"
    "is the only thing in the autograd path — so gradients flow through `mu`\n"
    "(linearly) and through `logsigma` (via the exp factor on eps).\n"
    "\n"
    "**Verification recipe.**\n"
    "1. Create `mu`, `logsigma` as leaf tensors with `requires_grad=True`.\n"
    "2. Reparameterize.\n"
    "3. `z.sum().backward()`.\n"
    "4. Assert `mu.grad is not None` (it should be `ones_like(mu)`).\n"
    "5. Assert `logsigma.grad is not None` (it should be `0.5 * eps * exp(0.5*logsigma)`)."
)


# ---------------------------------------------------------------------------
# SPEC 1 — discriminator-classifier-head ex2 (PatchGAN)
# ---------------------------------------------------------------------------

SPEC_PATCHGAN = {
    "atom_id": "discriminator-classifier-head",
    "subtopic": "GAN: Discriminator classifier head",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_PATCHGAN,
    "exercise_index": 2,
    "exercise_title": "PatchGAN-style classifier head (1x1 conv → per-patch sigmoid map)",
    "slug": "patchgan-1x1-conv-per-patch-sigmoid-head",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["patchgan", "1x1-conv", "sigmoid", "per-patch"],
    "kcs": [
        "1x1-conv-as-per-position-linear",
        "spatial-sigmoid-output-map",
    ],
    "lo": (
        "Apply a 1x1 `Conv2d(C_in, 1)` followed by `sigmoid` to convert a "
        "discriminator feature map into a per-patch real/fake probability "
        "map of shape `(B, 1, H, W)` — the PatchGAN deepening of the "
        "Flatten→Linear scalar head."
    ),
    "prompt_body": (
        "Implement `ex2_PatchHead`, an `nn.Module` for a PatchGAN-style "
        "discriminator head.\n\n"
        "`__init__(in_channels: int)` must:\n"
        "1. Call `super().__init__()`.\n"
        "2. Create `self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)`.\n\n"
        "`forward(x: Tensor) -> Tensor` where `x: (B, in_channels, H, W)`:\n"
        "1. Apply `self.conv` to get logits of shape `(B, 1, H, W)`.\n"
        "2. Return `t.sigmoid(logits)` — same shape, values in `(0, 1)`.\n\n"
        "**No flatten, no Linear, no avg-pool.** The whole point is to keep "
        "the spatial dimension."
    ),
    "stub": (
        "class ex2_PatchHead(nn.Module):\n"
        "    def __init__(self, in_channels: int):\n"
        '        """1x1-conv discriminator head producing a per-patch sigmoid map."""\n'
        "        raise NotImplementedError()\n"
        "\n"
        "    def forward(self, x: Tensor) -> Tensor:\n"
        "        raise NotImplementedError()"
    ),
    "test_body": (
        "# === Output shape preserves H, W; channel collapses to 1 ===\n"
        "head = ex2_PatchHead(in_channels=64)\n"
        "x = t.randn(4, 64, 8, 8)\n"
        "y = head(x)\n"
        "assert y.shape == (4, 1, 8, 8), f'expected (4,1,8,8), got {tuple(y.shape)}'\n"
        "\n"
        "# === Output values are valid sigmoid probabilities ===\n"
        "assert (y >= 0).all() and (y <= 1).all(), f'sigmoid output out of [0,1]: min={y.min():.4f}, max={y.max():.4f}'\n"
        "\n"
        "# === Works at a different spatial size ===\n"
        "head2 = ex2_PatchHead(in_channels=32)\n"
        "y2 = head2(t.randn(2, 32, 16, 16))\n"
        "assert y2.shape == (2, 1, 16, 16), f'expected (2,1,16,16), got {tuple(y2.shape)}'\n"
        "\n"
        "# === Only one Conv2d submodule, NO Linear, NO Flatten ===\n"
        "convs   = [m for m in head.modules() if isinstance(m, nn.Conv2d)]\n"
        "linears = [m for m in head.modules() if isinstance(m, nn.Linear)]\n"
        "flats   = [m for m in head.modules() if isinstance(m, nn.Flatten)]\n"
        "assert len(convs) == 1, f'expected exactly 1 Conv2d, got {len(convs)}'\n"
        "assert len(linears) == 0, f'PatchGAN head must NOT use Linear; got {len(linears)}'\n"
        "assert len(flats) == 0, f'PatchGAN head must NOT use Flatten; got {len(flats)}'\n"
        "\n"
        "# === Conv2d kernel is 1x1, in_channels matches, out_channels=1 ===\n"
        "conv = convs[0]\n"
        "assert conv.kernel_size == (1, 1), f'kernel must be 1x1, got {conv.kernel_size}'\n"
        "assert conv.in_channels == 64, f'in_channels must match constructor; got {conv.in_channels}'\n"
        "assert conv.out_channels == 1, f'out_channels must be 1; got {conv.out_channels}'\n"
        "\n"
        "# === Per-position equivalence: 1x1 conv == weight-tied Linear per spatial cell ===\n"
        "head3 = ex2_PatchHead(in_channels=8)\n"
        "x3 = t.randn(2, 8, 5, 5)\n"
        "y3 = head3(x3)\n"
        "# Manually compute via einsum + bias\n"
        "w = head3.conv.weight.view(1, 8)   # (1, 8) — the 1x1 conv kernel is a linear map\n"
        "b = head3.conv.bias                 # (1,)\n"
        "x3_flat = x3.permute(0, 2, 3, 1).reshape(-1, 8)   # (B*H*W, 8)\n"
        "y_manual_logits = x3_flat @ w.T + b               # (B*H*W, 1)\n"
        "y_manual = t.sigmoid(y_manual_logits).reshape(2, 5, 5, 1).permute(0, 3, 1, 2)\n"
        "assert t.allclose(y3, y_manual, atol=1e-5), 'PatchGAN head must equal per-position Linear+sigmoid'\n"
        "\n"
        "# === Parameter count matches a Linear(in_ch, 1): in_ch + 1 ===\n"
        "n_params = sum(p.numel() for p in head.parameters())\n"
        "assert n_params == 64 + 1, f'expected 65 params (64 weights + 1 bias), got {n_params}'"
    ),
    "solution_body": (
        "class ex2_PatchHead(nn.Module):\n"
        "    def __init__(self, in_channels: int):\n"
        "        super().__init__()\n"
        "        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)\n"
        "\n"
        "    def forward(self, x: Tensor) -> Tensor:\n"
        "        return t.sigmoid(self.conv(x))"
    ),
    "solution_notes": (
        "**1x1 conv = weight-tied Linear across spatial positions.** The "
        "(in_ch, 1, 1, 1) weight tensor reshapes to (1, in_ch) — exactly "
        "the same linear map applied independently at every (h, w) cell. "
        "Parameter count is `in_ch + 1` (weights + bias), independent of "
        "image size.\n\n"
        "**Sigmoid OUTSIDE the conv, not as part of it.** `nn.Conv2d` "
        "doesn't take an activation; you compose `Conv2d → sigmoid` "
        "explicitly. Same pattern as `BCEWithLogitsLoss` users skip the "
        "sigmoid — but for inference you want probabilities.\n\n"
        "**Why downstream loss changes too.** A PatchGAN trains with BCE "
        "against an `(B, 1, H, W)` target tensor (all-ones for real, all-"
        "zeros for fake). The reduction is `mean` over the patch dimension, "
        "so each patch contributes equally to the gradient."
    ),
    "extra_imports": EXTRA_NN_IMPORTS,
}


# ---------------------------------------------------------------------------
# SPEC 2 — elbo-loss-sum-with-beta ex2 (beta contrast)
# ---------------------------------------------------------------------------

SPEC_BETA_CONTRAST = {
    "atom_id": "elbo-loss-sum-with-beta",
    "subtopic": "VAE: ELBO loss sum with beta",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_BETA_CONTRAST,
    "exercise_index": 2,
    "exercise_title": "contrast beta in {0, 1, 4} on per-sample ELBO loss",
    "slug": "contrast-beta-zero-one-four-per-sample-elbo",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["beta-vae", "elbo", "per-sample", "kl-weight"],
    "kcs": [
        "per-sample-elbo-decomposition",
        "beta-monotone-in-loss-when-kl-positive",
    ],
    "lo": (
        "Analyze how the beta weight on the KL term affects per-sample ELBO "
        "loss by computing `recon + beta * kl` for `beta in {0, 1, 4}` on "
        "the same (x, x_hat, mu, logsigma) batch and verifying the "
        "non-decreasing-in-beta invariant per sample."
    ),
    "prompt_body": (
        "Implement `ex2_elbo_three_betas(x, x_hat, mu, logsigma)`.\n\n"
        "Inputs (all batched, same B):\n"
        "- `x:        (B, D)` — target.\n"
        "- `x_hat:    (B, D)` — VAE reconstruction.\n"
        "- `mu:       (B, latent)`\n"
        "- `logsigma: (B, latent)`\n\n"
        "Compute these per-sample tensors (each shape `(B,)`):\n\n"
        "1. `recon = ((x_hat - x) ** 2).flatten(1).sum(dim=1)` — squared-error "
        "reconstruction, summed over feature dims.\n"
        "2. `kl    = -0.5 * (1 + 2*logsigma - mu**2 - (2*logsigma).exp()).sum(dim=1)` — "
        "diagonal Gaussian KL vs N(0, I), per-sample.\n"
        "3. For each `beta in [0.0, 1.0, 4.0]`: `loss_beta = recon + beta * kl`.\n\n"
        "Return a dict:\n"
        "```\n"
        "{\n"
        "    'recon':      recon,        # (B,)\n"
        "    'kl':         kl,           # (B,)\n"
        "    'loss_beta0': loss_at_beta0,  # (B,)\n"
        "    'loss_beta1': loss_at_beta1,  # (B,)\n"
        "    'loss_beta4': loss_at_beta4,  # (B,)\n"
        "}\n"
        "```\n\n"
        "**No `.mean()` at the end.** Keep the per-sample vector."
    ),
    "stub": (
        "def ex2_elbo_three_betas(x: Tensor, x_hat: Tensor, mu: Tensor, logsigma: Tensor) -> dict:\n"
        '    """Per-sample ELBO loss at beta in {0, 1, 4}."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "B, D, L = 8, 16, 4\n"
        "x        = t.randn(B, D)\n"
        "x_hat    = x + 0.1 * t.randn(B, D)              # near-reconstruction\n"
        "mu       = t.randn(B, L) * 0.5\n"
        "logsigma = t.randn(B, L) * 0.3\n"
        "\n"
        "out = ex2_elbo_three_betas(x, x_hat, mu, logsigma)\n"
        "\n"
        "# === All keys present, all (B,) ===\n"
        "for k in ['recon', 'kl', 'loss_beta0', 'loss_beta1', 'loss_beta4']:\n"
        "    assert k in out, f'missing key {k!r}; got {list(out.keys())}'\n"
        "    v = out[k]\n"
        "    assert isinstance(v, Tensor), f'{k} must be a Tensor, got {type(v).__name__}'\n"
        "    assert v.shape == (B,), f'{k} must be (B,), got {tuple(v.shape)}'\n"
        "\n"
        "# === recon matches the spec exactly ===\n"
        "expected_recon = ((x_hat - x) ** 2).flatten(1).sum(dim=1)\n"
        "assert t.allclose(out['recon'], expected_recon, atol=1e-5), 'recon formula mismatch'\n"
        "\n"
        "# === kl matches the standard diagonal-Gaussian closed form ===\n"
        "expected_kl = -0.5 * (1 + 2*logsigma - mu**2 - (2*logsigma).exp()).sum(dim=1)\n"
        "assert t.allclose(out['kl'], expected_kl, atol=1e-5), 'kl formula mismatch'\n"
        "\n"
        "# === KL is non-negative per sample (Gaussian KL invariant) ===\n"
        "assert (out['kl'] >= -1e-5).all(), f'kl must be >= 0 per sample; got min={out[\"kl\"].min():.6f}'\n"
        "\n"
        "# === beta=0 ignores KL entirely; loss == recon ===\n"
        "assert t.allclose(out['loss_beta0'], out['recon'], atol=1e-5), 'loss at beta=0 must equal recon (KL dropped)'\n"
        "\n"
        "# === beta=1 standard ELBO ===\n"
        "assert t.allclose(out['loss_beta1'], out['recon'] + out['kl'], atol=1e-5)\n"
        "\n"
        "# === beta=4 = recon + 4*kl ===\n"
        "assert t.allclose(out['loss_beta4'], out['recon'] + 4.0 * out['kl'], atol=1e-5)\n"
        "\n"
        "# === Per-sample monotonicity: when KL > 0, loss is strictly increasing in beta ===\n"
        "kl = out['kl']\n"
        "pos = kl > 1e-6\n"
        "assert pos.any(), 'test fixture needs at least some samples with positive KL'\n"
        "assert (out['loss_beta1'][pos] > out['loss_beta0'][pos]).all(), 'loss_beta1 > loss_beta0 where kl>0'\n"
        "assert (out['loss_beta4'][pos] > out['loss_beta1'][pos]).all(), 'loss_beta4 > loss_beta1 where kl>0'\n"
        "\n"
        "# === Sanity: at mu=0, logsigma=0 (perfect prior match) → KL == 0 ===\n"
        "mu0 = t.zeros(B, L)\n"
        "ls0 = t.zeros(B, L)\n"
        "out2 = ex2_elbo_three_betas(x, x_hat, mu0, ls0)\n"
        "assert t.allclose(out2['kl'], t.zeros(B), atol=1e-5), f'mu=0, logsigma=0 must give kl=0; got max={out2[\"kl\"].max():.6f}'\n"
        "# When KL=0, all three losses agree exactly.\n"
        "assert t.allclose(out2['loss_beta0'], out2['loss_beta4'], atol=1e-5), 'KL=0 ⇒ beta has no effect'\n"
        "\n"
        "# === Higher-dim x (image-shaped) still flattens correctly ===\n"
        "x_img      = t.randn(B, 3, 4, 4)\n"
        "x_hat_img  = x_img + 0.05 * t.randn(B, 3, 4, 4)\n"
        "out3 = ex2_elbo_three_betas(x_img, x_hat_img, mu, logsigma)\n"
        "expected = ((x_hat_img - x_img) ** 2).flatten(1).sum(dim=1)\n"
        "assert t.allclose(out3['recon'], expected, atol=1e-5), 'recon must flatten image dims (B, C, H, W) → (B, C*H*W) before sum'"
    ),
    "solution_body": (
        "def ex2_elbo_three_betas(x, x_hat, mu, logsigma):\n"
        "    recon = ((x_hat - x) ** 2).flatten(1).sum(dim=1)\n"
        "    kl    = -0.5 * (1 + 2 * logsigma - mu ** 2 - (2 * logsigma).exp()).sum(dim=1)\n"
        "    return {\n"
        "        'recon':      recon,\n"
        "        'kl':         kl,\n"
        "        'loss_beta0': recon + 0.0 * kl,\n"
        "        'loss_beta1': recon + 1.0 * kl,\n"
        "        'loss_beta4': recon + 4.0 * kl,\n"
        "    }"
    ),
    "solution_notes": (
        "**Beta is monotone WHEN KL is positive.** Since KL >= 0 for any "
        "diagonal Gaussian, `loss(beta=b)` is non-decreasing in `b` per "
        "sample. Strict inequality requires strictly positive KL — at the "
        "prior `mu=0, logsigma=0`, all betas tie.\n\n"
        "**Why per-sample, not pre-reduced.** Returning `(B,)` lets the "
        "caller `.mean()` for the training step OR pick the worst-recon "
        "samples for inspection. Pre-reducing throws information away.\n\n"
        "**`flatten(1)` over `view(B, -1)`.** Equivalent but `flatten(1)` "
        "doesn't require knowing the batch size — useful when the batch "
        "dimension comes through a DataLoader with varying batch size at "
        "the tail."
    ),
    "extra_imports": EXTRA_NN_IMPORTS,
}


# ---------------------------------------------------------------------------
# SPEC 3 — generator-project-and-reshape ex2 (8x8 seed + 2 upsamples)
# ---------------------------------------------------------------------------

SPEC_8X8_PROJECT = {
    "atom_id": "generator-project-and-reshape",
    "subtopic": "GAN: Generator project + reshape",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_8X8_PROJECT,
    "exercise_index": 2,
    "exercise_title": "project latent → 8x8 seed → two ConvTranspose2d upsamples to 32x32",
    "slug": "project-to-8x8-then-two-upsamples-to-32x32",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["generator", "convtranspose2d", "upsample", "shapes"],
    "kcs": [
        "linear-project-then-view-to-spatial-seed",
        "convtranspose-2x-upsample-k4-s2-p1",
    ],
    "lo": (
        "Apply a `Linear(latent_dim, base_C * 8 * 8)` projection + `view` "
        "to produce an 8x8 spatial seed, then two `ConvTranspose2d(k=4, "
        "s=2, p=1)` upsamples to land at the requested 32x32 output, "
        "exposing each intermediate shape for verification."
    ),
    "prompt_body": (
        "Implement `ex2_Generator832`, an `nn.Module`. The deepening of "
        "ex1's 4x4 seed: now we use an **8x8** seed and TWO upsamples to "
        "reach **32x32**.\n\n"
        "`__init__(latent_dim: int, base_C: int = 64, out_C: int = 3)` "
        "creates:\n"
        "1. `self.project = nn.Linear(latent_dim, base_C * 8 * 8)`.\n"
        "2. `self.up1 = nn.ConvTranspose2d(base_C, base_C // 2, "
        "kernel_size=4, stride=2, padding=1)` — doubles spatial.\n"
        "3. `self.up2 = nn.ConvTranspose2d(base_C // 2, out_C, "
        "kernel_size=4, stride=2, padding=1)` — doubles again.\n"
        "Store `self.base_C` for the view step.\n\n"
        "`forward(z: Tensor) -> dict` where `z: (B, latent_dim)`:\n"
        "1. `h = self.project(z)` — shape `(B, base_C * 64)`.\n"
        "2. `seed = h.view(B, base_C, 8, 8)`.\n"
        "3. `mid = self.up1(seed)` — shape `(B, base_C // 2, 16, 16)`.\n"
        "4. `out = self.up2(mid)` — shape `(B, out_C, 32, 32)`.\n"
        "5. Return `{'seed': seed, 'mid': mid, 'out': out}` (intermediate "
        "shapes are the testable thing).\n\n"
        "No activations needed between the upsamples for this drill — "
        "we're verifying the shape pipeline, not the full DCGAN block."
    ),
    "stub": (
        "class ex2_Generator832(nn.Module):\n"
        "    def __init__(self, latent_dim: int, base_C: int = 64, out_C: int = 3):\n"
        '        """Latent → 8x8 spatial seed → 2 ConvTranspose2d upsamples to 32x32."""\n'
        "        raise NotImplementedError()\n"
        "\n"
        "    def forward(self, z: Tensor) -> dict:\n"
        "        raise NotImplementedError()"
    ),
    "test_body": (
        "# === End-to-end shape: (B, latent) → (B, out_C, 32, 32) ===\n"
        "gen = ex2_Generator832(latent_dim=128, base_C=64, out_C=3)\n"
        "z = t.randn(4, 128)\n"
        "out = gen(z)\n"
        "assert isinstance(out, dict), f'forward must return a dict, got {type(out).__name__}'\n"
        "assert set(out.keys()) == {'seed', 'mid', 'out'}, f'keys must be seed/mid/out, got {list(out.keys())}'\n"
        "\n"
        "# === Intermediate shapes match the pipeline ===\n"
        "assert out['seed'].shape == (4, 64, 8, 8),  f'seed must be (B, base_C, 8, 8); got {tuple(out[\"seed\"].shape)}'\n"
        "assert out['mid'].shape  == (4, 32, 16, 16),f'mid must be (B, base_C//2, 16, 16); got {tuple(out[\"mid\"].shape)}'\n"
        "assert out['out'].shape  == (4, 3, 32, 32), f'out must be (B, out_C, 32, 32); got {tuple(out[\"out\"].shape)}'\n"
        "\n"
        "# === Different base_C and out_C still work ===\n"
        "gen2 = ex2_Generator832(latent_dim=64, base_C=128, out_C=1)\n"
        "out2 = gen2(t.randn(2, 64))\n"
        "assert out2['seed'].shape == (2, 128, 8, 8)\n"
        "assert out2['mid'].shape  == (2, 64, 16, 16)\n"
        "assert out2['out'].shape  == (2, 1, 32, 32)\n"
        "\n"
        "# === Linear projection has correct shape: in=latent_dim, out=base_C*64 ===\n"
        "assert gen.project.in_features  == 128\n"
        "assert gen.project.out_features == 64 * 8 * 8, f'project out_features must be base_C * 64 = 4096; got {gen.project.out_features}'\n"
        "\n"
        "# === Both ConvTranspose2d use kernel_size=4, stride=2, padding=1 ===\n"
        "for name, m in [('up1', gen.up1), ('up2', gen.up2)]:\n"
        "    assert isinstance(m, nn.ConvTranspose2d), f'{name} must be ConvTranspose2d'\n"
        "    assert m.kernel_size == (4, 4), f'{name} kernel must be 4x4; got {m.kernel_size}'\n"
        "    assert m.stride      == (2, 2), f'{name} stride must be 2; got {m.stride}'\n"
        "    assert m.padding     == (1, 1), f'{name} padding must be 1; got {m.padding}'\n"
        "\n"
        "# === Channel pipeline: base_C → base_C//2 → out_C ===\n"
        "assert gen.up1.in_channels  == 64 and gen.up1.out_channels == 32\n"
        "assert gen.up2.in_channels  == 32 and gen.up2.out_channels == 3\n"
        "\n"
        "# === seed view is contiguous with project's output (same data, just reshaped) ===\n"
        "z_simple = t.zeros(3, 128)\n"
        "h_expected = gen.project(z_simple).view(3, 64, 8, 8)\n"
        "out_simple = gen(z_simple)\n"
        "assert t.allclose(out_simple['seed'], h_expected, atol=1e-6), 'seed must be project(z).view(B, base_C, 8, 8)'\n"
        "\n"
        "# === Spatial doubling invariant: each upsample exactly 2x in H AND W ===\n"
        "assert out['mid'].shape[-2] == 2 * out['seed'].shape[-2]\n"
        "assert out['mid'].shape[-1] == 2 * out['seed'].shape[-1]\n"
        "assert out['out'].shape[-2] == 2 * out['mid'].shape[-2]\n"
        "assert out['out'].shape[-1] == 2 * out['mid'].shape[-1]"
    ),
    "solution_body": (
        "class ex2_Generator832(nn.Module):\n"
        "    def __init__(self, latent_dim: int, base_C: int = 64, out_C: int = 3):\n"
        "        super().__init__()\n"
        "        self.base_C  = base_C\n"
        "        self.project = nn.Linear(latent_dim, base_C * 8 * 8)\n"
        "        self.up1     = nn.ConvTranspose2d(base_C, base_C // 2, kernel_size=4, stride=2, padding=1)\n"
        "        self.up2     = nn.ConvTranspose2d(base_C // 2, out_C, kernel_size=4, stride=2, padding=1)\n"
        "\n"
        "    def forward(self, z: Tensor) -> dict:\n"
        "        B = z.shape[0]\n"
        "        h    = self.project(z)\n"
        "        seed = h.view(B, self.base_C, 8, 8)\n"
        "        mid  = self.up1(seed)\n"
        "        out  = self.up2(mid)\n"
        "        return {'seed': seed, 'mid': mid, 'out': out}"
    ),
    "solution_notes": (
        "**`k=4, s=2, p=1` exactly doubles spatial.** Output size formula "
        "for ConvTranspose2d is `H_out = (H_in - 1)*s - 2*p + k`. With "
        "`s=2, p=1, k=4`: `H_out = 2*H_in - 2 + 4 - 2 = 2*H_in`. Clean "
        "doubling — the canonical DCGAN upsample block.\n\n"
        "**Why an 8x8 seed instead of 4x4.** With `final_H = 32` and the "
        "canonical 2x upsamples, you need `log2(32/seed_H)` upsamples. "
        "`seed=8` → 2 upsamples; `seed=4` → 3 upsamples. Larger seed = "
        "shallower decoder = easier to train at the cost of more Linear "
        "parameters in the projection.\n\n"
        "**`base_C` lives on `self` because forward needs it.** You could "
        "infer it from `self.project.out_features // 64` instead — same "
        "result, slightly less clear. Storing it explicit is the standard "
        "practice in ARENA / DCGAN code."
    ),
    "extra_imports": EXTRA_NN_IMPORTS,
}


# ---------------------------------------------------------------------------
# SPEC 4 — kl-divergence-gaussian-closed-form ex2 (scalar vs per-dim sum)
# ---------------------------------------------------------------------------

SPEC_KL_TWO_FORMS = {
    "atom_id": "kl-divergence-gaussian-closed-form",
    "subtopic": "VAE: KL divergence Gaussian closed-form",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_KL_TWO_FORMS,
    "exercise_index": 2,
    "exercise_title": "scalar 1-d KL closed form vs per-dim sum + batch mean — same number",
    "slug": "scalar-vs-per-dim-sum-batch-mean-kl",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["kl-divergence", "gaussian", "closed-form", "equivalence"],
    "kcs": [
        "scalar-kl-closed-form-from-mu-logsigma",
        "per-dim-sum-equals-elementwise-sum",
    ],
    "lo": (
        "Analyze the equivalence between the scalar 1-D Gaussian KL formula "
        "applied element-wise and the per-dim-sum-then-batch-mean form by "
        "computing both on the same `(mu, logsigma)` tensor and verifying "
        "they agree numerically to machine precision."
    ),
    "prompt_body": (
        "Implement `ex2_kl_two_forms(mu, logsigma)` where `mu: (B, D)` and "
        "`logsigma: (B, D)`.\n\n"
        "Compute BOTH forms of the diagonal-Gaussian KL vs `N(0, I)`:\n\n"
        "**Form A — element-wise scalar closed form, shape `(B, D)`:**\n"
        "```\n"
        "kl_elem[b, d] = 0.5 * (mu[b, d]**2 + (2 * logsigma[b, d]).exp() - 1 - 2 * logsigma[b, d])\n"
        "```\n"
        "\n"
        "**Form B — per-sample KL (sum over dims), shape `(B,)`:**\n"
        "```\n"
        "kl_per_sample[b] = -0.5 * sum_d(1 + 2*logsigma[b,d] - mu[b,d]**2 - exp(2*logsigma[b,d]))\n"
        "```\n"
        "\n"
        "Then compute:\n"
        "- `kl_batch_mean = kl_per_sample.mean()` — scalar.\n"
        "- `kl_from_elem  = kl_elem.sum(dim=1)`  — `(B,)`, should equal Form B per sample.\n\n"
        "Return:\n"
        "```\n"
        "{\n"
        "    'kl_elem':         kl_elem,         # (B, D)\n"
        "    'kl_per_sample':   kl_per_sample,   # (B,)\n"
        "    'kl_from_elem':    kl_from_elem,    # (B,)\n"
        "    'kl_batch_mean':   kl_batch_mean,   # scalar tensor\n"
        "}\n"
        "```\n"
    ),
    "stub": (
        "def ex2_kl_two_forms(mu: Tensor, logsigma: Tensor) -> dict:\n"
        '    """Compute scalar element-wise KL and per-sample sum + batch mean."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "B, D = 6, 4\n"
        "t.manual_seed(42)\n"
        "mu       = t.randn(B, D)\n"
        "logsigma = t.randn(B, D) * 0.5\n"
        "\n"
        "out = ex2_kl_two_forms(mu, logsigma)\n"
        "\n"
        "# === Keys + shapes ===\n"
        "assert set(out.keys()) == {'kl_elem', 'kl_per_sample', 'kl_from_elem', 'kl_batch_mean'}\n"
        "assert out['kl_elem'].shape       == (B, D)\n"
        "assert out['kl_per_sample'].shape == (B,)\n"
        "assert out['kl_from_elem'].shape  == (B,)\n"
        "assert out['kl_batch_mean'].dim() == 0, 'kl_batch_mean must be a scalar tensor'\n"
        "\n"
        "# === Form A: element-wise closed form matches the spec ===\n"
        "expected_elem = 0.5 * (mu**2 + (2 * logsigma).exp() - 1 - 2 * logsigma)\n"
        "assert t.allclose(out['kl_elem'], expected_elem, atol=1e-6), 'kl_elem element-wise formula mismatch'\n"
        "\n"
        "# === Form B: per-sample sum matches the spec ===\n"
        "expected_per = -0.5 * (1 + 2*logsigma - mu**2 - (2*logsigma).exp()).sum(dim=1)\n"
        "assert t.allclose(out['kl_per_sample'], expected_per, atol=1e-6), 'kl_per_sample sum-form mismatch'\n"
        "\n"
        "# === A and B agree: kl_elem.sum(dim=1) == kl_per_sample ===\n"
        "assert t.allclose(out['kl_from_elem'], out['kl_per_sample'], atol=1e-6), (\n"
        "    'element-wise sum across dims must equal per-sample sum form'\n"
        ")\n"
        "\n"
        "# === Both forms are non-negative (Gaussian KL invariant) ===\n"
        "assert (out['kl_elem'] >= -1e-6).all(),       f'element-wise KL must be >= 0; min={out[\"kl_elem\"].min():.6f}'\n"
        "assert (out['kl_per_sample'] >= -1e-6).all(), f'per-sample KL must be >= 0; min={out[\"kl_per_sample\"].min():.6f}'\n"
        "\n"
        "# === Sanity: mu=0, logsigma=0 → KL=0 everywhere ===\n"
        "out0 = ex2_kl_two_forms(t.zeros(B, D), t.zeros(B, D))\n"
        "assert t.allclose(out0['kl_elem'],       t.zeros(B, D), atol=1e-6)\n"
        "assert t.allclose(out0['kl_per_sample'], t.zeros(B),    atol=1e-6)\n"
        "assert out0['kl_batch_mean'].item() == 0.0\n"
        "\n"
        "# === Batch mean matches kl_per_sample.mean() ===\n"
        "assert t.allclose(out['kl_batch_mean'], out['kl_per_sample'].mean(), atol=1e-6)\n"
        "\n"
        "# === 1-D case (D=1): per-sample KL == kl_elem[:, 0] ===\n"
        "mu1       = t.randn(B, 1)\n"
        "logsigma1 = t.randn(B, 1) * 0.3\n"
        "out1d = ex2_kl_two_forms(mu1, logsigma1)\n"
        "assert t.allclose(out1d['kl_per_sample'], out1d['kl_elem'][:, 0], atol=1e-6), (\n"
        "    'with D=1, per-sample KL collapses to the single element of kl_elem'\n"
        ")\n"
        "\n"
        "# === KL grows quadratically in mu (at logsigma=0): KL(mu=2) ≈ 4 * KL(mu=1) per dim ===\n"
        "for ratio_check in [(1.0, 2.0, 4.0), (0.5, 1.0, 4.0)]:\n"
        "    m_small, m_big, expected_ratio = ratio_check\n"
        "    a = ex2_kl_two_forms(t.full((1, 1), m_small), t.zeros(1, 1))['kl_elem'].item()\n"
        "    b = ex2_kl_two_forms(t.full((1, 1), m_big),   t.zeros(1, 1))['kl_elem'].item()\n"
        "    assert abs(b / a - expected_ratio) < 1e-4, (\n"
        "        f'at logsigma=0, KL(mu={m_big})/KL(mu={m_small}) should be {expected_ratio}; got {b/a:.4f}'\n"
        "    )"
    ),
    "solution_body": (
        "def ex2_kl_two_forms(mu, logsigma):\n"
        "    # Form A: element-wise scalar closed form, shape (B, D).\n"
        "    kl_elem = 0.5 * (mu ** 2 + (2 * logsigma).exp() - 1 - 2 * logsigma)\n"
        "    # Form B: per-sample sum, shape (B,).\n"
        "    kl_per_sample = -0.5 * (1 + 2 * logsigma - mu ** 2 - (2 * logsigma).exp()).sum(dim=1)\n"
        "    kl_from_elem  = kl_elem.sum(dim=1)\n"
        "    kl_batch_mean = kl_per_sample.mean()\n"
        "    return {\n"
        "        'kl_elem':       kl_elem,\n"
        "        'kl_per_sample': kl_per_sample,\n"
        "        'kl_from_elem':  kl_from_elem,\n"
        "        'kl_batch_mean': kl_batch_mean,\n"
        "    }"
    ),
    "solution_notes": (
        "**The two forms are algebraically identical.** Distribute the "
        "negative half through Form B's parenthesis:\n"
        "`-0.5 * (1 + 2*logsigma - mu^2 - exp(2*logsigma))`\n"
        "`= -0.5 - logsigma + 0.5*mu^2 + 0.5*exp(2*logsigma)`\n"
        "`= 0.5*(mu^2 + exp(2*logsigma) - 1 - 2*logsigma)`,\n"
        "which is exactly Form A. Summing Form A over `dim=1` recovers "
        "Form B's per-sample value.\n\n"
        "**`(2 * logsigma).exp()` over `logsigma.exp() ** 2`.** Same result, "
        "but the multiplied-then-exp form is numerically more stable for "
        "large `|logsigma|` and one fewer floating-point op.\n\n"
        "**Why both forms are useful in practice.** `kl_elem` (the (B, D) "
        "tensor) shows you WHICH latent dims are doing the regularization "
        "work — collapsed dims have near-zero KL. `kl_per_sample` is the "
        "per-example value that goes into the loss. Both are computed "
        "cheaply from the same primitives."
    ),
    "extra_imports": EXTRA_NN_IMPORTS,
}


# ---------------------------------------------------------------------------
# SPEC 5 — mu-logsigma-encoder-head ex2 (two-head variant)
# ---------------------------------------------------------------------------

SPEC_TWO_HEAD = {
    "atom_id": "mu-logsigma-encoder-head",
    "subtopic": "VAE: mu+logsigma encoder head",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_TWO_HEAD,
    "exercise_index": 2,
    "exercise_title": "two-head encoder (separate Linears for mu and logsigma) — equivalent to single-Linear+chunk",
    "slug": "two-head-encoder-equivalent-to-single-linear-chunk",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["encoder", "two-head", "mu-logsigma", "weight-copy"],
    "kcs": [
        "two-separate-linears-mu-logsigma",
        "weight-copy-from-single-linear-row-blocks",
    ],
    "lo": (
        "Apply a two-head encoder (`fc_mu`, `fc_logsigma`) and verify it is "
        "mathematically equivalent to a single `Linear(d_in, 2*latent)` + "
        "`chunk(2, dim=-1)` by copying row-block weights from the single-"
        "head form into the two-head module."
    ),
    "prompt_body": (
        "Implement `ex2_TwoHeadEncoder`, an `nn.Module`.\n\n"
        "`__init__(d_in: int, latent_dim: int)` must:\n"
        "1. `super().__init__()`.\n"
        "2. `self.fc_mu       = nn.Linear(d_in, latent_dim)`.\n"
        "3. `self.fc_logsigma = nn.Linear(d_in, latent_dim)`.\n\n"
        "`forward(h: Tensor) -> tuple[Tensor, Tensor]` where `h: (B, d_in)`:\n"
        "- Return `(self.fc_mu(h), self.fc_logsigma(h))` — both `(B, "
        "latent_dim)`.\n\n"
        "Also implement `ex2_copy_from_single_head(two_head, single_linear)` "
        "which:\n"
        "1. Takes a `two_head: ex2_TwoHeadEncoder` and a `single_linear: "
        "nn.Linear(d_in, 2 * latent_dim)`.\n"
        "2. Copies row-block `[0:latent_dim]` of `single_linear.weight` "
        "into `two_head.fc_mu.weight` (and same slice of `bias` into "
        "`fc_mu.bias`).\n"
        "3. Copies row-block `[latent_dim:]` of `single_linear.weight` "
        "into `two_head.fc_logsigma.weight` (and `bias`).\n"
        "4. Uses `.data.copy_()` (in-place, no autograd tracking).\n"
        "5. Returns nothing (mutation).\n\n"
        "After copying, the two-head forward must produce the same `(mu, "
        "logsigma)` as `single_linear(h).chunk(2, dim=-1)` — that IS the "
        "equivalence test."
    ),
    "stub": (
        "class ex2_TwoHeadEncoder(nn.Module):\n"
        "    def __init__(self, d_in: int, latent_dim: int):\n"
        '        """Two separate Linears for mu and logsigma — VAE encoder head."""\n'
        "        raise NotImplementedError()\n"
        "\n"
        "    def forward(self, h: Tensor):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "def ex2_copy_from_single_head(two_head, single_linear):\n"
        '    """Copy row-blocks of single_linear into the two heads (in place)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "d_in, latent = 16, 4\n"
        "B = 5\n"
        "h = t.randn(B, d_in)\n"
        "\n"
        "# === Shapes from forward ===\n"
        "two_head = ex2_TwoHeadEncoder(d_in, latent)\n"
        "mu, logsigma = two_head(h)\n"
        "assert mu.shape       == (B, latent), f'mu must be (B, latent); got {tuple(mu.shape)}'\n"
        "assert logsigma.shape == (B, latent), f'logsigma must be (B, latent); got {tuple(logsigma.shape)}'\n"
        "\n"
        "# === Two SEPARATE Linears, both (d_in -> latent) ===\n"
        "linears = [m for m in two_head.modules() if isinstance(m, nn.Linear)]\n"
        "assert len(linears) == 2, f'expected exactly 2 Linear submodules, got {len(linears)}'\n"
        "for L in linears:\n"
        "    assert L.in_features  == d_in,   f'each Linear in_features must be d_in={d_in}; got {L.in_features}'\n"
        "    assert L.out_features == latent, f'each Linear out_features must be latent={latent}; got {L.out_features}'\n"
        "\n"
        "# === Both fc_mu and fc_logsigma exist by name ===\n"
        "assert hasattr(two_head, 'fc_mu')       and isinstance(two_head.fc_mu, nn.Linear)\n"
        "assert hasattr(two_head, 'fc_logsigma') and isinstance(two_head.fc_logsigma, nn.Linear)\n"
        "\n"
        "# === Parameter count == single Linear(d_in, 2*latent) param count ===\n"
        "n_th = sum(p.numel() for p in two_head.parameters())\n"
        "single = nn.Linear(d_in, 2 * latent)\n"
        "n_sh = sum(p.numel() for p in single.parameters())\n"
        "assert n_th == n_sh, f'two-head param count ({n_th}) must equal single-head Linear(d_in, 2*latent) ({n_sh})'\n"
        "\n"
        "# === Weight-copy equivalence: after copying row-blocks, forwards agree ===\n"
        "t.manual_seed(99)\n"
        "single = nn.Linear(d_in, 2 * latent)\n"
        "two_head2 = ex2_TwoHeadEncoder(d_in, latent)\n"
        "ex2_copy_from_single_head(two_head2, single)\n"
        "\n"
        "# Forward through both forms.\n"
        "mu_th, logsigma_th = two_head2(h)\n"
        "y_single = single(h)\n"
        "mu_sh, logsigma_sh = y_single.chunk(2, dim=-1)\n"
        "\n"
        "assert t.allclose(mu_th, mu_sh, atol=1e-6),             f'mu mismatch after copy; max-diff = {(mu_th - mu_sh).abs().max():.2e}'\n"
        "assert t.allclose(logsigma_th, logsigma_sh, atol=1e-6), f'logsigma mismatch after copy; max-diff = {(logsigma_th - logsigma_sh).abs().max():.2e}'\n"
        "\n"
        "# === Copied weights are independent from source after copy (no alias) ===\n"
        "assert two_head2.fc_mu.weight.data_ptr() != single.weight.data_ptr(), 'copy must NOT alias source weights'\n"
        "# Mutating the source after copying should NOT change the two-head output.\n"
        "single.weight.data.zero_()\n"
        "mu_th2, _ = two_head2(h)\n"
        "assert t.allclose(mu_th2, mu_th, atol=1e-6), 'two-head must NOT change after source weights are mutated'\n"
        "\n"
        "# === forward returns a tuple, not a Tensor or dict ===\n"
        "ret = two_head(h)\n"
        "assert isinstance(ret, tuple), f'forward must return a tuple, got {type(ret).__name__}'\n"
        "assert len(ret) == 2, f'tuple must have exactly 2 elements, got {len(ret)}'"
    ),
    "solution_body": (
        "class ex2_TwoHeadEncoder(nn.Module):\n"
        "    def __init__(self, d_in: int, latent_dim: int):\n"
        "        super().__init__()\n"
        "        self.fc_mu       = nn.Linear(d_in, latent_dim)\n"
        "        self.fc_logsigma = nn.Linear(d_in, latent_dim)\n"
        "\n"
        "    def forward(self, h):\n"
        "        return self.fc_mu(h), self.fc_logsigma(h)\n"
        "\n"
        "def ex2_copy_from_single_head(two_head, single_linear):\n"
        "    latent = two_head.fc_mu.out_features\n"
        "    two_head.fc_mu.weight.data.copy_(single_linear.weight.data[:latent])\n"
        "    two_head.fc_mu.bias.data.copy_(single_linear.bias.data[:latent])\n"
        "    two_head.fc_logsigma.weight.data.copy_(single_linear.weight.data[latent:])\n"
        "    two_head.fc_logsigma.bias.data.copy_(single_linear.bias.data[latent:])"
    ),
    "solution_notes": (
        "**The two forms are computationally equivalent.** A single "
        "`Linear(d_in, 2*latent)` is one matmul of shape `(B, d_in) @ "
        "(d_in, 2*latent)`. The two-head form is two matmuls of shape `(B, "
        "d_in) @ (d_in, latent)`. Same total FLOPs; same total params.\n\n"
        "**Why `.data.copy_()` over assignment.** Reassigning "
        "`two_head.fc_mu.weight = nn.Parameter(...)` would replace the "
        "Parameter object — the optimizer's reference to the old "
        "Parameter would dangle. `.data.copy_()` mutates the storage "
        "in-place, preserving Parameter identity.\n\n"
        "**Row-block slicing direction.** `nn.Linear.weight` has shape "
        "`(out_features, in_features)`. The first `latent` rows correspond "
        "to the first `latent` outputs (`y[:, :latent]`), which is what "
        "`chunk(2, dim=-1)[0]` returns. So `weight[:latent]` → `fc_mu`."
    ),
    "extra_imports": EXTRA_NN_IMPORTS,
}


# ---------------------------------------------------------------------------
# SPEC 6 — reparameterization-trick ex2 (gradient flow check)
# ---------------------------------------------------------------------------

SPEC_GRADIENT_CHECK = {
    "atom_id": "reparameterization-trick",
    "subtopic": "VAE: Reparameterization trick",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_GRADIENT_CHECK,
    "exercise_index": 2,
    "exercise_title": "gradient flow check — backward populates grads on BOTH mu and logsigma",
    "slug": "reparam-gradient-flows-to-mu-and-logsigma",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["reparameterization", "autograd", "gradient", "backward"],
    "kcs": [
        "reparam-puts-stochasticity-outside-graph",
        "backward-populates-grad-on-both-distribution-params",
    ],
    "lo": (
        "Analyze gradient flow through the reparameterization trick: given "
        "leaf tensors `mu` and `logsigma` with `requires_grad=True`, the "
        "function `ex2_reparam_check` reparameterizes a sample, calls "
        "`backward` on `z.sum()`, and verifies that BOTH `mu.grad` and "
        "`logsigma.grad` are populated with the analytically expected "
        "values."
    ),
    "prompt_body": (
        "Implement `ex2_reparam_check(mu, logsigma, eps)`.\n\n"
        "Inputs:\n"
        "- `mu:       (B, L)` — leaf tensor with `requires_grad=True`.\n"
        "- `logsigma: (B, L)` — leaf tensor with `requires_grad=True`.\n"
        "- `eps:      (B, L)` — pre-sampled noise from `N(0, 1)`. Already "
        "detached (no grad). The function does NOT sample its own noise — "
        "tests need determinism.\n\n"
        "Steps inside the function:\n"
        "1. Reparameterize: `z = mu + (0.5 * logsigma).exp() * eps`. Same "
        "as `mu + sigma * eps` where `sigma = exp(0.5 * logsigma)`.\n"
        "2. Compute `loss = z.sum()`.\n"
        "3. Call `loss.backward()`.\n"
        "4. Return a dict:\n"
        "   ```\n"
        "   {\n"
        "       'z':            z.detach(),         # (B, L)\n"
        "       'mu_grad':      mu.grad.clone(),    # (B, L) — should be ones\n"
        "       'logsigma_grad': logsigma.grad.clone(),  # (B, L) — should be 0.5 * eps * exp(0.5*logsigma)\n"
        "   }\n"
        "   ```\n\n"
        "**Don't zero gradients before calling.** Tests pre-zero them. Don't "
        "wrap in `no_grad`. The whole point is to LET autograd build the "
        "graph."
    ),
    "stub": (
        "def ex2_reparam_check(mu: Tensor, logsigma: Tensor, eps: Tensor) -> dict:\n"
        '    """Reparameterize, backward on z.sum(), return grads on mu and logsigma."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "B, L = 4, 3\n"
        "t.manual_seed(7)\n"
        "mu_val       = t.randn(B, L)\n"
        "logsigma_val = t.randn(B, L) * 0.3\n"
        "eps          = t.randn(B, L)  # caller-provided noise — already detached\n"
        "\n"
        "mu       = mu_val.clone().requires_grad_(True)\n"
        "logsigma = logsigma_val.clone().requires_grad_(True)\n"
        "\n"
        "out = ex2_reparam_check(mu, logsigma, eps)\n"
        "\n"
        "# === Keys + shapes ===\n"
        "assert set(out.keys()) == {'z', 'mu_grad', 'logsigma_grad'}\n"
        "assert out['z'].shape             == (B, L)\n"
        "assert out['mu_grad'].shape       == (B, L)\n"
        "assert out['logsigma_grad'].shape == (B, L)\n"
        "\n"
        "# === z = mu + exp(0.5*logsigma) * eps  (forward correctness) ===\n"
        "expected_z = mu_val + (0.5 * logsigma_val).exp() * eps\n"
        "assert t.allclose(out['z'], expected_z, atol=1e-6), 'reparam formula mismatch'\n"
        "\n"
        "# === Grad on mu must be EXACTLY ones (z = mu + ..., d(z.sum())/d(mu) = 1) ===\n"
        "expected_mu_grad = t.ones(B, L)\n"
        "assert t.allclose(out['mu_grad'], expected_mu_grad, atol=1e-6), (\n"
        "    f'mu.grad must be ones; got max-diff = {(out[\"mu_grad\"] - expected_mu_grad).abs().max():.2e}'\n"
        ")\n"
        "\n"
        "# === Grad on logsigma is 0.5 * eps * exp(0.5*logsigma) ===\n"
        "# z = mu + exp(0.5*logsigma) * eps\n"
        "# dz/dlogsigma = exp(0.5*logsigma) * eps * 0.5\n"
        "# d(z.sum())/dlogsigma_{b,l} = same per element\n"
        "expected_logsigma_grad = 0.5 * eps * (0.5 * logsigma_val).exp()\n"
        "assert t.allclose(out['logsigma_grad'], expected_logsigma_grad, atol=1e-6), (\n"
        "    f'logsigma.grad must equal 0.5 * eps * exp(0.5*logsigma); '\n"
        "    f'max-diff = {(out[\"logsigma_grad\"] - expected_logsigma_grad).abs().max():.2e}'\n"
        ")\n"
        "\n"
        "# === Neither grad is None (the failure mode if reparam is bypassed) ===\n"
        "assert out['mu_grad']       is not None\n"
        "assert out['logsigma_grad'] is not None\n"
        "\n"
        "# === eps had no grad to start with and gets no grad after backward ===\n"
        "assert eps.grad is None, 'eps should have no grad — it is the noise source, not a parameter'\n"
        "\n"
        "# === When logsigma is very negative, sigma is small, so logsigma_grad ≈ 0 ===\n"
        "# Bound: |grad| = 0.5 * |eps| * exp(0.5*logsigma). With logsigma=-30 → sigma≈3.1e-7, grad ~ 1.5e-7 per element.\n"
        "mu2       = t.zeros(2, 2, requires_grad=True)\n"
        "logsigma2 = t.full((2, 2), -30.0, requires_grad=True)\n"
        "eps2      = t.randn(2, 2)\n"
        "out2 = ex2_reparam_check(mu2, logsigma2, eps2)\n"
        "assert out2['logsigma_grad'].abs().max() < 1e-5, (\n"
        "    f'with logsigma=-30, sigma is tiny so logsigma.grad must be near zero; got {out2[\"logsigma_grad\"].abs().max():.2e}'\n"
        ")\n"
        "# But mu.grad is still exactly 1.\n"
        "assert t.allclose(out2['mu_grad'], t.ones(2, 2), atol=1e-6)\n"
        "\n"
        "# === eps = 0 forces logsigma_grad to exactly 0 (since dz/dlogsigma scales with eps) ===\n"
        "mu3       = t.randn(3, 3, requires_grad=True)\n"
        "logsigma3 = t.randn(3, 3, requires_grad=True)\n"
        "eps3      = t.zeros(3, 3)\n"
        "out3 = ex2_reparam_check(mu3, logsigma3, eps3)\n"
        "assert t.allclose(out3['logsigma_grad'], t.zeros(3, 3), atol=1e-7), (\n"
        "    'when eps=0, sigma cancels out → no gradient on logsigma'\n"
        ")\n"
        "# mu.grad still ones.\n"
        "assert t.allclose(out3['mu_grad'], t.ones(3, 3), atol=1e-6)"
    ),
    "solution_body": (
        "def ex2_reparam_check(mu, logsigma, eps):\n"
        "    sigma = (0.5 * logsigma).exp()\n"
        "    z = mu + sigma * eps\n"
        "    loss = z.sum()\n"
        "    loss.backward()\n"
        "    return {\n"
        "        'z':             z.detach(),\n"
        "        'mu_grad':       mu.grad.clone(),\n"
        "        'logsigma_grad': logsigma.grad.clone(),\n"
        "    }"
    ),
    "solution_notes": (
        "**Why mu.grad is exactly 1.** `z = mu + (sigma * eps)`. The `mu` "
        "branch enters linearly, so `dz/dmu = 1` element-wise. Backprop on "
        "`z.sum()` adds these element-wise grads — every entry of "
        "`mu.grad` is exactly 1.\n\n"
        "**Why logsigma.grad scales with eps.** `dz/dlogsigma = "
        "(d/dlogsigma)[exp(0.5*logsigma) * eps] = 0.5 * exp(0.5*logsigma) "
        "* eps`. When `eps` is large the gradient through the variance "
        "branch is large; when `eps = 0` the gradient is exactly zero — "
        "the model can't learn `logsigma` from samples where the noise "
        "didn't kick.\n\n"
        "**The non-reparameterized failure mode.** If you'd written "
        "`z = t.distributions.Normal(mu, sigma).sample()`, the `.sample()` "
        "call cuts the autograd graph — `z.requires_grad` would be False, "
        "and `backward` would either raise or silently leave both grads as "
        "None. The reparam trick exists precisely to keep the graph "
        "intact.\n\n"
        "**`mu.grad.clone()` over `mu.grad`.** A subsequent backward call "
        "in the caller's code could accumulate INTO `mu.grad`. Returning a "
        "clone prevents downstream side-effects from mutating our returned "
        "values."
    ),
    "extra_imports": EXTRA_NN_IMPORTS,
}


# ---------------------------------------------------------------------------
# All specs
# ---------------------------------------------------------------------------

SPECS = [
    SPEC_PATCHGAN,
    SPEC_BETA_CONTRAST,
    SPEC_8X8_PROJECT,
    SPEC_KL_TWO_FORMS,
    SPEC_TWO_HEAD,
    SPEC_GRADIENT_CHECK,
]


# ---------------------------------------------------------------------------
# Verifier
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
    print(f"[deepening_ab_batch13] Verifying {len(SPECS)} specs...")
    _verify_all(SPECS)

    print(f"\n[deepening_ab_batch13] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_ab_batch13] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
