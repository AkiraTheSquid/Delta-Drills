#!/usr/bin/env python3
"""Author Colab-native standalones for ARENA chapter-0 part-5 (VAE / GAN) atoms.

Batch 6: eight single-exercise drills for component atoms inside `prereqs_vae_gan/`.
Each atom gets ONE ex (ex1). Each exercise hits ONE Bloom level + at most 2 KCs.

Atom roster (8):
    - mu-logsigma-encoder-head           (Linear -> chunk(2, dim=-1) -> mu + logsigma)
    - reparameterization-trick           (mu + (0.5 * logsigma).exp() * eps, eps=randn_like)
    - kl-divergence-gaussian-closed-form (-0.5 * sum(1 + logsigma - mu^2 - exp(logsigma)))
    - elbo-loss-sum-with-beta            (loss = recon + beta * kl)
    - generator-project-and-reshape      ((B,100) -> Linear(100,1024*4*4) -> (B,1024,4,4))
    - convtranspose-bn-activation-block  (nn.Sequential(ConvT, BN, ReLU), 4x4 -> 8x8)
    - conv-leakyrelu-block-discriminator (nn.Sequential(Conv2d s=2, BN, LeakyReLU(0.2)), 32 -> 16)
    - discriminator-classifier-head      (flatten + Linear(features, 1) + sigmoid)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_vae_gan"


# ---------------------------------------------------------------- recaps

RECAP_MU_LOGSIGMA = (
    "## mu + logsigma encoder head — quick refresher\n"
    "\n"
    "A VAE encoder ends in TWO heads — one for `mu`, one for `logsigma` — "
    "both of shape `(B, latent_dim)`. The cheapest implementation is a "
    "SINGLE `nn.Linear(D, 2 * latent_dim)` followed by `x.chunk(2, dim=-1)`:\n"
    "\n"
    "```python\n"
    "params = self.head(features)            # (B, 2 * latent_dim)\n"
    "mu, logsigma = params.chunk(2, dim=-1)  # each (B, latent_dim)\n"
    "```\n"
    "\n"
    "**Why `logsigma`, not `sigma`.** A neural net output is unconstrained — "
    "it can be negative. Predicting `logsigma` and exponentiating later "
    "(`sigma = logsigma.exp()`) guarantees `sigma > 0` without any clamping. "
    "Predicting `sigma` directly would need a softplus or an abs, which "
    "behave badly near zero.\n"
    "\n"
    "**Compared to two separate `Linear` heads.** Functionally identical "
    "(both are `(D -> latent_dim)` affine maps). The single-head + chunk "
    "form saves one parameter object and reads more clearly as `'encoder "
    "emits the parameters of a Gaussian'`. ARENA uses this form."
)

RECAP_REPARAM = (
    "## Reparameterization trick — quick refresher\n"
    "\n"
    "The trick that makes VAEs trainable. Instead of sampling `z ~ N(mu, "
    "sigma^2)` directly (which is non-differentiable w.r.t. `mu` and "
    "`sigma`), we sample `eps ~ N(0, 1)` and compute:\n"
    "\n"
    "```python\n"
    "sigma = (0.5 * logsigma).exp()   # sigma = exp(logsigma / 2)\n"
    "z = mu + sigma * eps             # eps = randn_like(mu)\n"
    "```\n"
    "\n"
    "Now gradients flow through `mu` (direct) and `sigma` (via the "
    "multiplicative path), and the randomness lives in `eps` — a constant "
    "from autograd's point of view.\n"
    "\n"
    "**Why `0.5 * logsigma`.** The encoder emits log-VARIANCE, not "
    "log-standard-deviation. `sigma = sqrt(var) = sqrt(exp(logvar)) = "
    "exp(logvar / 2)`. The factor of 0.5 in the exponent IS the square root.\n"
    "\n"
    "**Why `randn_like(mu)`.** Matches shape, dtype, and device "
    "automatically. `t.randn(*mu.shape)` would default to `float32` on CPU "
    "— breaks silently when `mu` is on GPU or `bfloat16`."
)

RECAP_KL = (
    "## KL divergence (Gaussian, closed-form) — quick refresher\n"
    "\n"
    "For a Gaussian posterior `q(z|x) = N(mu, sigma^2)` vs the standard-"
    "normal prior `p(z) = N(0, 1)`, the KL divergence has a closed form:\n"
    "\n"
    "```\n"
    "KL(q || p) = -0.5 * sum(1 + logsigma - mu^2 - exp(logsigma))\n"
    "```\n"
    "\n"
    "where `logsigma` is the encoder's log-VARIANCE output (yes — the "
    "ARENA convention uses `logsigma` as the variable name even though it "
    "is treating it as log-var; the formula below matches that).\n"
    "\n"
    "**Sum over latent dim, mean over batch.** Sum across `latent_dim` to "
    "get per-sample KL `(B,)`. Mean across batch (or sum, scaled later) "
    "to get the scalar loss term:\n"
    "```python\n"
    "kl_per_sample = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=-1)\n"
    "kl_scalar = kl_per_sample.mean()\n"
    "```\n"
    "\n"
    "**Sanity check.** When `mu=0` and `logsigma=0` (i.e. `sigma=1`), the "
    "posterior == prior and `KL == 0`. Easy unit test: build a zero "
    "tensor, run your function, assert near zero."
)

RECAP_ELBO = (
    "## ELBO loss (sum with beta) — quick refresher\n"
    "\n"
    "The VAE training objective decomposes into TWO terms:\n"
    "\n"
    "```\n"
    "loss = reconstruction + beta * kl\n"
    "```\n"
    "\n"
    "- `reconstruction` — e.g. `F.mse_loss(decoded, original)` or a "
    "Bernoulli cross-entropy. Drives the decoder to actually decode.\n"
    "- `kl` — the closed-form Gaussian KL term. Pulls the latent "
    "distribution toward the standard-normal prior.\n"
    "- `beta` — a positive scalar trading reconstruction quality against "
    "latent regularity.\n"
    "\n"
    "**`beta = 1` is the vanilla VAE.** Maximizes the true ELBO. "
    "**`beta > 1` is the beta-VAE** (Higgins et al. 2017) — encourages "
    "DISENTANGLED latents at the cost of reconstruction. **`beta < 1` "
    "tilts toward reconstruction** at the cost of latent regularity — "
    "useful when you don't care about sampling from the prior.\n"
    "\n"
    "**Both terms are SCALARS before you combine them.** Reduce each to "
    "a 0-D tensor (mean or sum over batch+dims), THEN add. Adding a "
    "`(B,)` to a `()` will broadcast — silent bug."
)

RECAP_GEN_PROJ = (
    "## Generator project + reshape — quick refresher\n"
    "\n"
    "The first layer of a DCGAN generator turns a flat noise vector "
    "`z (B, 100)` into a spatial seed tensor `(B, 1024, 4, 4)` — the "
    "smallest feature map that subsequent ConvTranspose layers will "
    "upsample.\n"
    "\n"
    "```python\n"
    "self.proj = nn.Linear(latent_dim, 1024 * 4 * 4)  # bias optional\n"
    "...\n"
    "x = self.proj(z)                  # (B, 16384)\n"
    "x = x.view(B, 1024, 4, 4)         # spatial seed\n"
    "```\n"
    "\n"
    "**Why `Linear` then `view`, not `ConvTranspose` from a 1×1 seed.** "
    "Either works, but `Linear + view` is what the original DCGAN paper "
    "(Radford et al.) does — and it's faster (single matmul, no padding "
    "math). The result is identical.\n"
    "\n"
    "**Shape arithmetic.** From `(B, 1024, 4, 4)` four ConvTranspose "
    "stride-2 blocks produce `4 -> 8 -> 16 -> 32 -> 64` — final image "
    "size 64×64. The channel count halves each time: `1024 -> 512 -> "
    "256 -> 128 -> 3`."
)

RECAP_CONVT_BN_ACT = (
    "## ConvTranspose + BN + activation block — quick refresher\n"
    "\n"
    "The repeated unit of a DCGAN generator. Three layers stacked in "
    "`nn.Sequential`:\n"
    "\n"
    "```python\n"
    "nn.Sequential(\n"
    "    nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),\n"
    "    nn.BatchNorm2d(out_c),\n"
    "    nn.ReLU(inplace=True),\n"
    ")\n"
    "```\n"
    "\n"
    "**Stride 2, kernel 4, padding 1 doubles spatial size.** Output "
    "shape `H_out = (H_in - 1) * stride - 2 * padding + kernel = 2 * H_in`. "
    "So `4 -> 8 -> 16 -> 32 -> 64` for a 4-block stack.\n"
    "\n"
    "**`bias=False`** because BatchNorm immediately re-centres the "
    "feature map — the ConvTranspose bias is redundant and just wastes "
    "parameters. Standard DCGAN convention.\n"
    "\n"
    "**ReLU (not LeakyReLU) on the GENERATOR.** Discriminator uses "
    "LeakyReLU; generator uses ReLU. The asymmetry is from the original "
    "paper — the generator needs sharp activations to produce crisp "
    "outputs, the discriminator needs a gentle slope on negatives to "
    "avoid dying neurons."
)

RECAP_CONV_LR_DISC = (
    "## Conv + BN + LeakyReLU discriminator block — quick refresher\n"
    "\n"
    "The repeated unit of a DCGAN discriminator. Three layers in "
    "`nn.Sequential`, downsampling by 2 every block:\n"
    "\n"
    "```python\n"
    "nn.Sequential(\n"
    "    nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),\n"
    "    nn.BatchNorm2d(out_c),\n"
    "    nn.LeakyReLU(0.2, inplace=True),\n"
    ")\n"
    "```\n"
    "\n"
    "**Stride 2, kernel 4, padding 1 halves spatial size.** Output "
    "`H_out = (H_in + 2*padding - kernel) // stride + 1 = H_in // 2`. "
    "So `32 -> 16 -> 8 -> 4 -> 1` for a 4-block stack on 32×32 input.\n"
    "\n"
    "**LeakyReLU(0.2) — the discriminator default.** Slope 0.2 on the "
    "negative side keeps gradient flowing for inputs the discriminator "
    "currently thinks are fake. Plain ReLU would zero those gradients — "
    "the discriminator would stop learning from negative examples.\n"
    "\n"
    "**First block usually skips BN.** ARENA's implementation has the "
    "FIRST conv block of the discriminator omit BatchNorm — adding it on "
    "the raw RGB input scales the image stats away. All INTERMEDIATE "
    "blocks include BN."
)

RECAP_DISC_HEAD = (
    "## Discriminator classifier head — quick refresher\n"
    "\n"
    "The last layer of a DCGAN discriminator turns a `(B, C, H, W)` "
    "feature map into a scalar probability per sample — the answer to "
    "'is this real?'.\n"
    "\n"
    "```python\n"
    "x = features.flatten(start_dim=1)   # (B, C*H*W)\n"
    "logits = self.classifier(x)         # (B, 1)\n"
    "probs = t.sigmoid(logits).squeeze(-1)  # (B,)\n"
    "```\n"
    "\n"
    "**Sigmoid, not softmax.** Real-vs-fake is BINARY — there is exactly "
    "one positive class. Sigmoid produces `P(real)` directly; "
    "`P(fake) = 1 - P(real)`. Softmax over a single output is degenerate.\n"
    "\n"
    "**Numerical note.** In practice you'd return the LOGITS and pair "
    "them with `F.binary_cross_entropy_with_logits` for the loss — that "
    "fuses the sigmoid with the BCE for numerical stability. The "
    "explicit `sigmoid` here is for INFERENCE / inspection.\n"
    "\n"
    "**Flatten before the Linear.** A `(B, 1024, 4, 4)` feature map has "
    "`B * 16384` numbers. The classifier is `nn.Linear(16384, 1)` — "
    "expects a 2-D input. `.flatten(start_dim=1)` collapses the channel "
    "and spatial axes into one feature axis without touching the batch."
)


# ---------------------------------------------------------------- specs

SPECS = []

# -------------------------- ex1 / mu-logsigma-encoder-head
SPECS.append({
    "atom_id": "mu-logsigma-encoder-head",
    "subtopic": "VAE: mu+logsigma encoder head",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_MU_LOGSIGMA,
    "exercise_index": 1,
    "exercise_title": "single-head Linear + chunk into mu and logsigma",
    "slug": "single-head-linear-chunk-mu-logsigma",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["vae", "encoder", "chunk", "gaussian-head"],
    "kcs": ["encoder-double-width-linear", "chunk-split-mu-logsigma"],
    "lo": (
        "Apply a single `nn.Linear(D, 2*latent_dim)` followed by `chunk(2, "
        "dim=-1)` to project encoder features into the Gaussian parameters "
        "`(mu, logsigma)`, each of shape `(B, latent_dim)`."
    ),
    "prompt_body": (
        "Implement `ex1_encoder_head(features, weight, bias, latent_dim)`. The "
        "double-width-Linear-plus-chunk pattern every VAE encoder uses:\n\n"
        "1. `features` has shape `(B, D)` — output of the encoder body, just "
        "before the Gaussian-parameter heads.\n"
        "2. `weight` has shape `(2 * latent_dim, D)` and `bias` has shape "
        "`(2 * latent_dim,)` — together they parameterize an `nn.Linear(D, "
        "2 * latent_dim)`.\n"
        "3. Compute the affine: `params = features @ weight.T + bias` → "
        "shape `(B, 2 * latent_dim)`.\n"
        "4. Split with `params.chunk(2, dim=-1)` into `mu` and `logsigma`, "
        "each `(B, latent_dim)`.\n"
        "5. Return the tuple `(mu, logsigma)`.\n\n"
        "Input: `features` `(B, D)`, `weight` `(2*latent_dim, D)`, `bias` "
        "`(2*latent_dim,)`, `latent_dim` `int`.\n"
        "Output: tuple `((B, latent_dim), (B, latent_dim))`.\n\n"
        "The visualization runs your head on random features for "
        "`latent_dim=8` and renders the per-batch mu and logsigma as two "
        "side-by-side heatmaps — useful for spotting dead latent dims at a "
        "glance."
    ),
    "stub": (
        "def ex1_encoder_head(features: Tensor, weight: Tensor, bias: Tensor, latent_dim: int) -> tuple[Tensor, Tensor]:\n"
        '    """Affine to 2*latent_dim, chunk into (mu, logsigma)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Smoke test — known shapes.\n"
        "B, D, latent_dim = 4, 32, 5\n"
        "rng = t.Generator().manual_seed(0)\n"
        "feats = t.randn(B, D, generator=rng)\n"
        "W = t.randn(2 * latent_dim, D, generator=rng)\n"
        "b = t.randn(2 * latent_dim, generator=rng)\n"
        "mu, logsigma = ex1_encoder_head(feats, W, b, latent_dim)\n"
        "assert mu.shape == (B, latent_dim), f'mu shape wrong: {tuple(mu.shape)}'\n"
        "assert logsigma.shape == (B, latent_dim), f'logsigma shape wrong: {tuple(logsigma.shape)}'\n"
        "assert mu.dtype == t.float32 and logsigma.dtype == t.float32\n"
        "\n"
        "# Numerical correctness — chunk must split the FIRST half into mu, second half into logsigma.\n"
        "expected_full = feats @ W.T + b   # (B, 2*latent_dim)\n"
        "expected_mu = expected_full[:, :latent_dim]\n"
        "expected_logsigma = expected_full[:, latent_dim:]\n"
        "assert t.allclose(mu, expected_mu, atol=1e-5), 'mu must be first half of affine'\n"
        "assert t.allclose(logsigma, expected_logsigma, atol=1e-5), 'logsigma must be second half of affine'\n"
        "\n"
        "# Zero-input: with bias only, mu == bias[:latent_dim], logsigma == bias[latent_dim:].\n"
        "zero_feats = t.zeros(2, D)\n"
        "zmu, zls = ex1_encoder_head(zero_feats, W, b, latent_dim)\n"
        "assert t.allclose(zmu[0], b[:latent_dim], atol=1e-5)\n"
        "assert t.allclose(zls[0], b[latent_dim:], atol=1e-5)\n"
        "\n"
        "# --- Visualization: per-batch mu and logsigma heatmaps ---\n"
        "B_viz, D_viz, latent_viz = 16, 64, 8\n"
        "rng = t.Generator().manual_seed(11)\n"
        "feats_viz = t.randn(B_viz, D_viz, generator=rng)\n"
        "W_viz = 0.3 * t.randn(2 * latent_viz, D_viz, generator=rng)\n"
        "b_viz = t.zeros(2 * latent_viz)\n"
        "mu_viz, ls_viz = ex1_encoder_head(feats_viz, W_viz, b_viz, latent_viz)\n"
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))\n"
        "im1 = ax1.imshow(mu_viz.numpy(), aspect='auto', cmap='RdBu_r')\n"
        "ax1.set_title('mu (B × latent_dim)')\n"
        "ax1.set_xlabel('latent dim'); ax1.set_ylabel('batch idx')\n"
        "plt.colorbar(im1, ax=ax1, fraction=0.046)\n"
        "im2 = ax2.imshow(ls_viz.numpy(), aspect='auto', cmap='RdBu_r')\n"
        "ax2.set_title('logsigma (B × latent_dim)')\n"
        "ax2.set_xlabel('latent dim'); ax2.set_ylabel('batch idx')\n"
        "plt.colorbar(im2, ax=ax2, fraction=0.046)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_encoder_head(features: Tensor, weight: Tensor, bias: Tensor, latent_dim: int) -> tuple[Tensor, Tensor]:\n"
        "    params = features @ weight.T + bias   # (B, 2 * latent_dim)\n"
        "    mu, logsigma = params.chunk(2, dim=-1)\n"
        "    return mu, logsigma"
    ),
    "solution_notes": (
        "**`chunk(2, dim=-1)` vs slicing.** `params.chunk(2, dim=-1)` is "
        "equivalent to `(params[..., :latent_dim], params[..., latent_dim:])` "
        "— but it doesn't hard-code the split point, and reads as 'split in "
        "two along the last axis'. If you ever change `latent_dim`, the "
        "slicing breaks; the chunk doesn't.\n\n"
        "**Why one Linear, not two.** Functionally identical, but the "
        "single-head form is one parameter object and one matmul instead of "
        "two — easier to checkpoint, faster on hardware. ARENA's reference "
        "VAE uses this convention.\n\n"
        "**The heatmap is the diagnostic.** A dead latent dim shows up as a "
        "vertical stripe of constant logsigma near 0 (the encoder has given "
        "up on that dim). A healthy VAE shows varied magnitudes across dims "
        "and samples."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
})


# -------------------------- ex1 / reparameterization-trick
SPECS.append({
    "atom_id": "reparameterization-trick",
    "subtopic": "VAE: Reparameterization trick",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_REPARAM,
    "exercise_index": 1,
    "exercise_title": "differentiable Gaussian sampling with gradient check",
    "slug": "reparameterized-gaussian-sample",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["reparam", "randn_like", "differentiable-sample", "vae"],
    "kcs": ["reparam-formula", "randn-like-noise"],
    "lo": (
        "Apply the reparameterization trick — `z = mu + (0.5 * logsigma).exp() "
        "* eps` with `eps = randn_like(mu)` — to draw a differentiable "
        "Gaussian sample from `N(mu, exp(logsigma))`."
    ),
    "prompt_body": (
        "Implement `ex1_reparameterize(mu, logsigma)`. The differentiable-"
        "sample trick that makes VAEs trainable end-to-end:\n\n"
        "1. `mu` and `logsigma` both have shape `(B, latent_dim)` — outputs "
        "of the VAE encoder head.\n"
        "2. Compute `sigma = (0.5 * logsigma).exp()` — i.e. "
        "`exp(logsigma / 2)`. (Treat `logsigma` as log-variance — the ARENA "
        "convention.)\n"
        "3. Draw noise: `eps = t.randn_like(mu)`. Use `randn_like` (not "
        "`t.randn(*mu.shape)`) so dtype + device match `mu` automatically.\n"
        "4. Return `z = mu + sigma * eps` — same shape as `mu`.\n\n"
        "Input: `mu`, `logsigma` — `(B, latent_dim)` float tensors.\n"
        "Output: `(B, latent_dim)` float tensor.\n\n"
        "The visualization compares the empirical distribution of 2-D "
        "samples to the theoretical mean ± 2σ ellipse for several "
        "`(mu, logsigma)` choices."
    ),
    "stub": (
        "def ex1_reparameterize(mu: Tensor, logsigma: Tensor) -> Tensor:\n"
        '    """z = mu + exp(logsigma / 2) * eps, eps ~ N(0, 1)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Shape / dtype smoke test.\n"
        "B, latent_dim = 8, 4\n"
        "mu = t.zeros(B, latent_dim)\n"
        "logsigma = t.zeros(B, latent_dim)\n"
        "t.manual_seed(0)\n"
        "z = ex1_reparameterize(mu, logsigma)\n"
        "assert z.shape == (B, latent_dim), f'expected (B,latent_dim), got {tuple(z.shape)}'\n"
        "assert z.dtype == t.float32\n"
        "\n"
        "# Distribution sanity — with mu=0, logsigma=0 → sigma=1 → z ~ N(0, 1).\n"
        "t.manual_seed(0)\n"
        "big_mu = t.zeros(20000, 3)\n"
        "big_ls = t.zeros(20000, 3)\n"
        "z_big = ex1_reparameterize(big_mu, big_ls)\n"
        "assert abs(z_big.mean().item()) < 0.05, f'mean should be ~0, got {z_big.mean().item():.4f}'\n"
        "assert abs(z_big.std().item() - 1.0) < 0.05, f'std should be ~1, got {z_big.std().item():.4f}'\n"
        "\n"
        "# Distribution sanity — with mu=5, logsigma=log(4) → sigma=2 → z ~ N(5, 4).\n"
        "import math\n"
        "t.manual_seed(0)\n"
        "mu_shift = t.full((20000, 3), 5.0)\n"
        "ls_shift = t.full((20000, 3), math.log(4.0))    # variance=4 → sigma=2\n"
        "z_shift = ex1_reparameterize(mu_shift, ls_shift)\n"
        "assert abs(z_shift.mean().item() - 5.0) < 0.1, f'mean should be ~5, got {z_shift.mean().item():.4f}'\n"
        "assert abs(z_shift.std().item() - 2.0) < 0.1, f'std should be ~2, got {z_shift.std().item():.4f}'\n"
        "\n"
        "# Differentiability — gradient must flow back to mu and logsigma.\n"
        "g_mu = t.zeros(2, 3, requires_grad=True)\n"
        "g_ls = t.zeros(2, 3, requires_grad=True)\n"
        "t.manual_seed(7)\n"
        "g_z = ex1_reparameterize(g_mu, g_ls)\n"
        "g_z.sum().backward()\n"
        "assert g_mu.grad is not None and (g_mu.grad != 0).any(), 'mu must receive nonzero gradient'\n"
        "assert g_ls.grad is not None, 'logsigma must receive gradient'\n"
        "\n"
        "# --- Visualization: empirical samples vs theoretical mean ± 2σ ellipse ---\n"
        "fig, axes = plt.subplots(1, 3, figsize=(12, 4))\n"
        "cases = [\n"
        "    (t.tensor([0.0, 0.0]), t.tensor([0.0, 0.0]),       'mu=(0,0), sigma=(1,1)'),\n"
        "    (t.tensor([2.0, -1.0]), t.tensor([math.log(0.25), math.log(1.0)]), 'mu=(2,-1), sigma=(0.5,1)'),\n"
        "    (t.tensor([-1.0, 1.5]), t.tensor([math.log(4.0), math.log(0.25)]), 'mu=(-1,1.5), sigma=(2,0.5)'),\n"
        "]\n"
        "for ax, (m, ls, title) in zip(axes, cases):\n"
        "    t.manual_seed(0)\n"
        "    mb = m.unsqueeze(0).expand(2000, -1)\n"
        "    lb = ls.unsqueeze(0).expand(2000, -1)\n"
        "    samples = ex1_reparameterize(mb, lb)\n"
        "    ax.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), s=4, alpha=0.4, color='steelblue')\n"
        "    # 2σ ellipse from theoretical params.\n"
        "    sig = (0.5 * ls).exp()\n"
        "    theta = t.linspace(0, 2 * math.pi, 100)\n"
        "    ex = m[0].item() + 2 * sig[0].item() * theta.cos().numpy()\n"
        "    ey = m[1].item() + 2 * sig[1].item() * theta.sin().numpy()\n"
        "    ax.plot(ex, ey, 'r-', lw=2, label='theoretical 2σ')\n"
        "    ax.scatter([m[0].item()], [m[1].item()], color='red', marker='x', s=80, label='theoretical mean')\n"
        "    ax.set_title(title)\n"
        "    ax.set_aspect('equal')\n"
        "    ax.legend(fontsize=8)\n"
        "    ax.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_reparameterize(mu: Tensor, logsigma: Tensor) -> Tensor:\n"
        "    sigma = (0.5 * logsigma).exp()\n"
        "    eps = t.randn_like(mu)\n"
        "    return mu + sigma * eps"
    ),
    "solution_notes": (
        "**Why the trick works.** Sampling `z ~ N(mu, sigma^2)` directly has "
        "no gradient — you can't differentiate through a random draw. By "
        "moving the randomness OUT of the path (into `eps`) and combining it "
        "deterministically with `mu` and `sigma`, the gradient flows through "
        "`mu` (the additive path) and `sigma` (the multiplicative path) — "
        "exactly the parameters we need to train.\n\n"
        "**Why `(0.5 * logsigma).exp()` and not `logsigma.exp() ** 0.5`.** "
        "Mathematically identical (`exp(x/2) = sqrt(exp(x))`), but the "
        "former is one fewer op and numerically friendlier — no square root "
        "of a possibly-tiny number.\n\n"
        "**`randn_like(mu)` is the right choice.** It matches dtype + device "
        "+ shape in one call. `t.randn(*mu.shape, device=mu.device, "
        "dtype=mu.dtype)` is correct but verbose; bare `t.randn(*mu.shape)` "
        "silently breaks on GPU."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
})


# -------------------------- ex1 / kl-divergence-gaussian-closed-form
SPECS.append({
    "atom_id": "kl-divergence-gaussian-closed-form",
    "subtopic": "VAE: KL divergence Gaussian closed-form",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_KL,
    "exercise_index": 1,
    "exercise_title": "closed-form Gaussian KL with per-sample bar chart",
    "slug": "gaussian-kl-closed-form",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["kl-divergence", "vae", "closed-form", "gaussian"],
    "kcs": ["kl-gaussian-formula", "kl-sum-then-mean"],
    "lo": (
        "Apply the closed-form Gaussian-vs-standard-normal KL `-0.5 * "
        "sum(1 + logsigma - mu^2 - exp(logsigma))` to compute (a) per-"
        "sample KL of shape `(B,)` and (b) the batch-mean scalar."
    ),
    "prompt_body": (
        "Implement `ex1_kl_gaussian(mu, logsigma)`. The closed-form KL term "
        "that completes the VAE ELBO:\n\n"
        "1. `mu` and `logsigma` both have shape `(B, latent_dim)` — the "
        "encoder's Gaussian-parameter outputs.\n"
        "2. Compute the per-element KL contribution:\n"
        "   `-0.5 * (1 + logsigma - mu**2 - exp(logsigma))` → "
        "`(B, latent_dim)`.\n"
        "3. Sum across `latent_dim` (the last axis) → per-sample KL `(B,)`.\n"
        "4. Mean across batch → scalar (0-D tensor).\n"
        "5. Return the tuple `(per_sample_kl, scalar_kl)`.\n\n"
        "Input: `mu`, `logsigma` — `(B, latent_dim)` float tensors.\n"
        "Output: tuple `((B,) tensor, scalar tensor)`.\n\n"
        "The visualization bar-charts the per-sample KL on a synthetic "
        "batch where some samples have `mu` and `logsigma` near zero (low "
        "KL, posterior ≈ prior) and others are far from zero (high KL)."
    ),
    "stub": (
        "def ex1_kl_gaussian(mu: Tensor, logsigma: Tensor) -> tuple[Tensor, Tensor]:\n"
        '    """Return (per-sample KL of shape (B,), scalar batch mean)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import math\n"
        "\n"
        "# Posterior == prior case: mu=0, logsigma=0 → KL = 0.\n"
        "B, latent_dim = 4, 5\n"
        "mu_z = t.zeros(B, latent_dim)\n"
        "ls_z = t.zeros(B, latent_dim)\n"
        "per_sample, scalar = ex1_kl_gaussian(mu_z, ls_z)\n"
        "assert per_sample.shape == (B,), f'expected (B,), got {tuple(per_sample.shape)}'\n"
        "assert scalar.shape == (), f'expected scalar (), got {tuple(scalar.shape)}'\n"
        "assert t.allclose(per_sample, t.zeros(B), atol=1e-6), f'KL must be 0 at posterior=prior, got {per_sample}'\n"
        "assert t.allclose(scalar, t.tensor(0.0), atol=1e-6)\n"
        "\n"
        "# Closed-form check vs explicit per-dim computation.\n"
        "rng = t.Generator().manual_seed(0)\n"
        "mu = t.randn(3, 4, generator=rng)\n"
        "ls = t.randn(3, 4, generator=rng)\n"
        "expected_per_elem = -0.5 * (1 + ls - mu.pow(2) - ls.exp())\n"
        "expected_per_sample = expected_per_elem.sum(dim=-1)\n"
        "expected_scalar = expected_per_sample.mean()\n"
        "ps, sc = ex1_kl_gaussian(mu, ls)\n"
        "assert t.allclose(ps, expected_per_sample, atol=1e-5), 'per-sample KL formula mismatch'\n"
        "assert t.allclose(sc, expected_scalar, atol=1e-5), 'scalar KL must equal mean of per-sample'\n"
        "\n"
        "# Variance-only deviation: mu=0, logsigma != 0.\n"
        "# When mu=0 and logsigma=c, per-sample KL = latent_dim * (-0.5 * (1 + c - exp(c))).\n"
        "c = 0.5\n"
        "mu_var = t.zeros(2, 6)\n"
        "ls_var = t.full((2, 6), c)\n"
        "ps_var, _ = ex1_kl_gaussian(mu_var, ls_var)\n"
        "expected_kl = 6 * (-0.5 * (1 + c - math.exp(c)))\n"
        "assert t.allclose(ps_var, t.full((2,), expected_kl), atol=1e-5), (\n"
        "    f'variance-only KL mismatch: got {ps_var}, expected {expected_kl}'\n"
        ")\n"
        "\n"
        "# KL must be non-negative (mathematical fact).\n"
        "for _ in range(5):\n"
        "    test_mu = t.randn(5, 8)\n"
        "    test_ls = t.randn(5, 8)\n"
        "    p, _ = ex1_kl_gaussian(test_mu, test_ls)\n"
        "    assert (p >= -1e-5).all(), f'KL must be >= 0, got {p}'\n"
        "\n"
        "# --- Visualization: per-sample KL bar chart on a graded batch ---\n"
        "B_viz = 12\n"
        "# Samples 0..3 near prior; 4..7 medium drift; 8..11 large drift.\n"
        "drift_levels = t.cat([\n"
        "    t.full((4,), 0.0),\n"
        "    t.full((4,), 1.0),\n"
        "    t.full((4,), 3.0),\n"
        "])\n"
        "mu_viz = drift_levels.unsqueeze(-1).expand(B_viz, 6)\n"
        "ls_viz = (0.3 * drift_levels).unsqueeze(-1).expand(B_viz, 6)\n"
        "per_viz, _ = ex1_kl_gaussian(mu_viz, ls_viz)\n"
        "colors = ['steelblue'] * 4 + ['orange'] * 4 + ['crimson'] * 4\n"
        "fig, ax = plt.subplots(figsize=(8, 3.5))\n"
        "ax.bar(range(B_viz), per_viz.numpy(), color=colors, edgecolor='black')\n"
        "ax.set_xlabel('batch sample idx')\n"
        "ax.set_ylabel('per-sample KL')\n"
        "ax.set_title('ex1 per-sample KL — blue: posterior≈prior, orange: drift, red: far')\n"
        "ax.grid(True, alpha=0.3, axis='y')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_kl_gaussian(mu: Tensor, logsigma: Tensor) -> tuple[Tensor, Tensor]:\n"
        "    per_elem = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp())\n"
        "    per_sample = per_elem.sum(dim=-1)\n"
        "    scalar = per_sample.mean()\n"
        "    return per_sample, scalar"
    ),
    "solution_notes": (
        "**Where the closed form comes from.** For two Gaussians "
        "`q = N(mu, sigma^2)` and `p = N(0, 1)`, KL has a closed form:\n"
        "`KL(q || p) = 0.5 * (mu^2 + sigma^2 - 1 - log(sigma^2))`.\n"
        "Substituting `log(sigma^2) = logsigma` (ARENA convention treats "
        "`logsigma` as log-variance) and rearranging gives the formula "
        "you implemented.\n\n"
        "**Sum across latent, mean across batch.** Per-sample KL is the "
        "sum over latent dims because dims are INDEPENDENT under the "
        "diagonal-covariance assumption. Mean across batch makes the loss "
        "comparable across batch sizes.\n\n"
        "**KL is always ≥ 0.** A solid sanity test for your "
        "implementation. If you ever get negative KL, you've sign-flipped "
        "the `-0.5` or dropped a term."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
})


# -------------------------- ex1 / elbo-loss-sum-with-beta
SPECS.append({
    "atom_id": "elbo-loss-sum-with-beta",
    "subtopic": "VAE: ELBO loss sum with beta",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_ELBO,
    "exercise_index": 1,
    "exercise_title": "beta-weighted ELBO sum + sweep plot",
    "slug": "beta-elbo-sum",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["elbo", "beta-vae", "loss-composition", "scalar"],
    "kcs": ["elbo-sum-recon-plus-kl", "beta-weighting"],
    "lo": (
        "Apply the beta-weighted ELBO recipe `loss = reconstruction + beta * "
        "kl` to combine two pre-reduced scalar loss terms into a single "
        "scalar training loss."
    ),
    "prompt_body": (
        "Implement `ex1_elbo_loss(reconstruction, kl, beta)`. The "
        "one-line VAE loss combinator:\n\n"
        "1. `reconstruction` is a 0-D scalar tensor — the pre-reduced "
        "reconstruction loss (e.g. `F.mse_loss(decoded, original)`).\n"
        "2. `kl` is a 0-D scalar tensor — the batch-mean Gaussian KL.\n"
        "3. `beta` is a Python `float` — the KL weighting.\n"
        "4. Return `reconstruction + beta * kl` as a 0-D scalar tensor.\n\n"
        "Input: two scalar tensors + a float.\n"
        "Output: scalar tensor.\n\n"
        "The visualization sweeps `beta` from 0 to 4 on a fixed "
        "(reconstruction, kl) pair and plots how the composite loss scales "
        "— the slope IS `kl`, the intercept IS `reconstruction`."
    ),
    "stub": (
        "def ex1_elbo_loss(reconstruction: Tensor, kl: Tensor, beta: float) -> Tensor:\n"
        '    """Return reconstruction + beta * kl as a scalar tensor."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Smoke test — known values.\n"
        "recon = t.tensor(0.5)\n"
        "kl = t.tensor(2.0)\n"
        "loss = ex1_elbo_loss(recon, kl, beta=1.0)\n"
        "assert loss.shape == (), f'expected scalar (), got {tuple(loss.shape)}'\n"
        "assert t.allclose(loss, t.tensor(2.5)), f'expected 2.5, got {loss.item()}'\n"
        "\n"
        "# beta=0 → pure reconstruction.\n"
        "loss_b0 = ex1_elbo_loss(recon, kl, beta=0.0)\n"
        "assert t.allclose(loss_b0, recon), f'beta=0 → loss must equal reconstruction, got {loss_b0.item()}'\n"
        "\n"
        "# beta=4 (beta-VAE) → recon + 4*kl.\n"
        "loss_b4 = ex1_elbo_loss(recon, kl, beta=4.0)\n"
        "assert t.allclose(loss_b4, t.tensor(0.5 + 4 * 2.0)), f'beta=4 mismatch, got {loss_b4.item()}'\n"
        "\n"
        "# beta=0.1 (recon-tilted) → recon + 0.1*kl.\n"
        "loss_b01 = ex1_elbo_loss(recon, kl, beta=0.1)\n"
        "assert t.allclose(loss_b01, t.tensor(0.5 + 0.1 * 2.0)), f'beta=0.1 mismatch, got {loss_b01.item()}'\n"
        "\n"
        "# Differentiability — gradient must flow through both terms.\n"
        "r = t.tensor(1.0, requires_grad=True)\n"
        "k = t.tensor(0.5, requires_grad=True)\n"
        "L = ex1_elbo_loss(r, k, beta=2.0)\n"
        "L.backward()\n"
        "assert t.allclose(r.grad, t.tensor(1.0)), f'd loss / d recon must be 1, got {r.grad.item()}'\n"
        "assert t.allclose(k.grad, t.tensor(2.0)), f'd loss / d kl must be beta=2, got {k.grad.item()}'\n"
        "\n"
        "# --- Visualization: composite loss as a function of beta ---\n"
        "recon_fixed = t.tensor(1.0)\n"
        "kl_fixed = t.tensor(0.7)\n"
        "betas = t.linspace(0, 4, 50)\n"
        "losses = [ex1_elbo_loss(recon_fixed, kl_fixed, b.item()).item() for b in betas]\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "ax.plot(betas.numpy(), losses, color='purple', lw=2)\n"
        "ax.axhline(recon_fixed.item(), color='gray', ls='--', label=f'recon={recon_fixed.item()}')\n"
        "ax.axvline(1.0, color='black', ls=':', alpha=0.5, label='beta=1 (standard VAE)')\n"
        "ax.set_xlabel('beta')\n"
        "ax.set_ylabel('total ELBO loss')\n"
        "ax.set_title(f'ex1 loss = recon + beta * kl  (recon={recon_fixed.item()}, kl={kl_fixed.item()})')\n"
        "ax.legend()\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_elbo_loss(reconstruction: Tensor, kl: Tensor, beta: float) -> Tensor:\n"
        "    return reconstruction + beta * kl"
    ),
    "solution_notes": (
        "**Why `beta` is a float, not a tensor.** `beta` is a fixed "
        "hyperparameter — not a learned quantity. Passing it as a Python "
        "float saves wrapping/unwrapping and makes the call site read "
        "naturally: `loss = elbo(recon, kl, beta=4.0)`.\n\n"
        "**The two terms are already scalar.** Make sure your `reconstruction` "
        "and `kl` inputs are 0-D tensors BEFORE calling this. If `kl` were "
        "`(B,)`, broadcasting would add `kl` to `recon` elementwise and "
        "you'd silently get a `(B,)` 'loss' — which `.backward()` would "
        "accept (autograd sums it implicitly), but the per-sample average "
        "is wrong.\n\n"
        "**`beta=1` IS the true ELBO.** Anything else trades regularity "
        "for reconstruction (or vice versa) — useful, but no longer an "
        "ELBO. Higgins et al. (2017) showed `beta > 1` encourages "
        "disentangled latent factors on visual datasets."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
})


# -------------------------- ex1 / generator-project-and-reshape
SPECS.append({
    "atom_id": "generator-project-and-reshape",
    "subtopic": "GAN: Generator project + reshape",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_GEN_PROJ,
    "exercise_index": 1,
    "exercise_title": "latent-to-spatial seed projection + reshape",
    "slug": "latent-to-spatial-seed",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["gan", "generator", "linear", "view", "spatial-seed"],
    "kcs": ["latent-linear-projection", "view-to-spatial-seed"],
    "lo": (
        "Apply a `nn.Linear(latent_dim, C*H*W)` followed by `view(B, C, "
        "H, W)` to turn a flat noise vector `(B, 100)` into a spatial "
        "seed tensor `(B, 1024, 4, 4)` for a DCGAN generator."
    ),
    "prompt_body": (
        "Implement `ex1_project_and_reshape(z, weight, bias, channels, "
        "spatial)`. The first layer of every DCGAN generator:\n\n"
        "1. `z` has shape `(B, latent_dim)` — typically `latent_dim=100`.\n"
        "2. `weight` has shape `(channels * spatial * spatial, latent_dim)`, "
        "`bias` has shape `(channels * spatial * spatial,)`. Together they "
        "parameterize `nn.Linear(latent_dim, channels * spatial * spatial)`.\n"
        "3. Affine project: `flat = z @ weight.T + bias` → "
        "`(B, channels * spatial * spatial)`.\n"
        "4. Reshape to `(B, channels, spatial, spatial)` with "
        "`flat.view(B, channels, spatial, spatial)`.\n\n"
        "Input: `z` `(B, latent_dim)`; `weight`/`bias`; `channels` int; "
        "`spatial` int.\n"
        "Output: `(B, channels, spatial, spatial)` float tensor.\n\n"
        "The visualization renders the per-channel mean intensity of the "
        "first generated seed as a heatmap — useful sanity check that "
        "the spatial seed actually varies across channels."
    ),
    "stub": (
        "def ex1_project_and_reshape(z: Tensor, weight: Tensor, bias: Tensor, channels: int, spatial: int) -> Tensor:\n"
        '    """Project (B, latent) -> (B, C*H*W) and view as (B, C, H, W)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Canonical DCGAN shape: (B, 100) -> (B, 1024, 4, 4).\n"
        "B, latent_dim, C, H = 3, 100, 1024, 4\n"
        "rng = t.Generator().manual_seed(0)\n"
        "z = t.randn(B, latent_dim, generator=rng)\n"
        "W = 0.01 * t.randn(C * H * H, latent_dim, generator=rng)\n"
        "b = t.zeros(C * H * H)\n"
        "out = ex1_project_and_reshape(z, W, b, channels=C, spatial=H)\n"
        "assert out.shape == (B, C, H, H), f'expected ({B},{C},{H},{H}), got {tuple(out.shape)}'\n"
        "assert out.dtype == t.float32\n"
        "\n"
        "# Numerical correctness vs reference manual computation.\n"
        "expected_flat = z @ W.T + b\n"
        "expected = expected_flat.view(B, C, H, H)\n"
        "assert t.allclose(out, expected, atol=1e-5), 'output must equal (z @ W.T + b).view(...)'\n"
        "\n"
        "# Smaller shape — make sure it generalizes.\n"
        "B2, ld2, C2, H2 = 5, 32, 8, 2\n"
        "z2 = t.randn(B2, ld2)\n"
        "W2 = t.randn(C2 * H2 * H2, ld2)\n"
        "b2 = t.randn(C2 * H2 * H2)\n"
        "out2 = ex1_project_and_reshape(z2, W2, b2, channels=C2, spatial=H2)\n"
        "assert out2.shape == (B2, C2, H2, H2)\n"
        "\n"
        "# Bias-only case: z=0 → all batch items receive bias.view(C, H, H).\n"
        "zero_z = t.zeros(2, latent_dim)\n"
        "fixed_b = t.arange(C * H * H, dtype=t.float32)\n"
        "bias_out = ex1_project_and_reshape(zero_z, W, fixed_b, channels=C, spatial=H)\n"
        "assert t.allclose(bias_out[0], fixed_b.view(C, H, H), atol=1e-5), 'z=0 → output should be bias reshape'\n"
        "assert t.allclose(bias_out[0], bias_out[1], atol=1e-5), 'z=0 → all batch items identical'\n"
        "\n"
        "# --- Visualization: per-channel mean of the first generated seed ---\n"
        "rng = t.Generator().manual_seed(11)\n"
        "z_viz = t.randn(4, latent_dim, generator=rng)\n"
        "W_viz = 0.05 * t.randn(C * H * H, latent_dim, generator=rng)\n"
        "b_viz = t.zeros(C * H * H)\n"
        "seed = ex1_project_and_reshape(z_viz, W_viz, b_viz, channels=C, spatial=H)\n"
        "# Take the first sample, average each channel to a single number, render the (32, 32) grid of channel-means.\n"
        "ch_means = seed[0].mean(dim=[1, 2])  # (C,)\n"
        "grid = ch_means.view(32, 32)  # 1024 = 32 * 32 — nice square for viz\n"
        "fig, ax = plt.subplots(figsize=(5, 5))\n"
        "im = ax.imshow(grid.numpy(), cmap='viridis')\n"
        "ax.set_title(f'ex1 per-channel mean intensity\\n(sample 0 of seed (B={B}, C={C}, H={H}))')\n"
        "ax.set_xlabel('channel idx % 32')\n"
        "ax.set_ylabel('channel idx // 32')\n"
        "plt.colorbar(im, ax=ax, fraction=0.046)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_project_and_reshape(z: Tensor, weight: Tensor, bias: Tensor, channels: int, spatial: int) -> Tensor:\n"
        "    B = z.shape[0]\n"
        "    flat = z @ weight.T + bias                          # (B, C*H*W)\n"
        "    return flat.view(B, channels, spatial, spatial)     # (B, C, H, W)"
    ),
    "solution_notes": (
        "**Why `view`, not `reshape` or `rearrange`.** All three work on a "
        "contiguous tensor. `view` is the cheapest (no copy guarantee — "
        "throws if not contiguous, which here it always is). `reshape` is "
        "more permissive (silently copies if needed). `einops.rearrange("
        "flat, 'b (c h w) -> b c h w', c=channels, h=spatial)` is also "
        "fine and more readable.\n\n"
        "**Why `Linear + view`, not start from a `(B, 100, 1, 1)` "
        "ConvTranspose seed.** Functionally either lets you upsample to "
        "`64 × 64`. The DCGAN paper uses Linear + view because (a) it's "
        "one matmul vs an expensive transposed convolution from a 1×1 "
        "source, and (b) you have direct control over the output channels "
        "(`1024 * 4 * 4 = 16384` parameters per output unit, vs the "
        "constrained channel-count of a ConvTranspose from 1×1)."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
})


# -------------------------- ex1 / convtranspose-bn-activation-block
SPECS.append({
    "atom_id": "convtranspose-bn-activation-block",
    "subtopic": "GAN: ConvT+BN+Activation block",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_CONVT_BN_ACT,
    "exercise_index": 1,
    "exercise_title": "build a 4x4 to 8x8 generator upsampling block",
    "slug": "convtranspose-bn-relu-block",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["gan", "generator", "convtranspose", "batchnorm", "sequential"],
    "kcs": ["sequential-three-layer-block", "convtranspose-stride-doubles-spatial"],
    "lo": (
        "Apply `nn.Sequential` to wire `nn.ConvTranspose2d(stride=2, "
        "kernel=4, padding=1, bias=False) -> nn.BatchNorm2d -> nn.ReLU` "
        "into a single channel-halving, spatial-doubling generator block."
    ),
    "prompt_body": (
        "Implement `ex1_build_generator_block(in_channels, out_channels)`. "
        "The repeated unit of a DCGAN generator:\n\n"
        "1. Construct an `nn.Sequential` containing three layers IN ORDER:\n"
        "   - `nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, "
        "stride=2, padding=1, bias=False)`\n"
        "   - `nn.BatchNorm2d(out_channels)`\n"
        "   - `nn.ReLU(inplace=True)`\n"
        "2. `bias=False` on the ConvTranspose because BatchNorm immediately "
        "follows.\n"
        "3. Stride 2 + kernel 4 + padding 1 DOUBLES the spatial size — a "
        "4×4 input becomes 8×8 output.\n"
        "4. Return the Sequential.\n\n"
        "Input: `in_channels`, `out_channels` — ints.\n"
        "Output: `nn.Sequential` module.\n\n"
        "The visualization runs your block on a `(1, 1024, 4, 4)` seed "
        "and renders four output channel slices as a 2×2 grid of 8×8 "
        "feature maps."
    ),
    "stub": (
        "def ex1_build_generator_block(in_channels: int, out_channels: int) -> t.nn.Sequential:\n"
        '    """Return nn.Sequential(ConvTranspose2d, BatchNorm2d, ReLU)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "\n"
        "# Build a 4x4 -> 8x8 generator block (channel halving).\n"
        "block = ex1_build_generator_block(in_channels=1024, out_channels=512)\n"
        "assert isinstance(block, nn.Sequential), 'must return nn.Sequential'\n"
        "layers = list(block.children())\n"
        "assert len(layers) == 3, f'expected exactly 3 layers, got {len(layers)}'\n"
        "assert isinstance(layers[0], nn.ConvTranspose2d), f'layer 0 must be ConvTranspose2d, got {type(layers[0]).__name__}'\n"
        "assert isinstance(layers[1], nn.BatchNorm2d), f'layer 1 must be BatchNorm2d, got {type(layers[1]).__name__}'\n"
        "assert isinstance(layers[2], nn.ReLU), f'layer 2 must be ReLU, got {type(layers[2]).__name__}'\n"
        "\n"
        "# ConvTranspose configuration.\n"
        "ct = layers[0]\n"
        "assert ct.in_channels == 1024 and ct.out_channels == 512, f'ConvT channels wrong: in={ct.in_channels}, out={ct.out_channels}'\n"
        "assert ct.kernel_size == (4, 4), f'kernel must be 4, got {ct.kernel_size}'\n"
        "assert ct.stride == (2, 2), f'stride must be 2, got {ct.stride}'\n"
        "assert ct.padding == (0, 0) or ct.padding == (1, 1), f'padding should be 1, got {ct.padding}'\n"
        "assert ct.padding == (1, 1), f'padding must be 1 for spatial-doubling, got {ct.padding}'\n"
        "assert ct.bias is None, 'bias must be False (BatchNorm follows)'\n"
        "\n"
        "# BatchNorm configuration.\n"
        "bn = layers[1]\n"
        "assert bn.num_features == 512, f'BatchNorm num_features must match ConvT out, got {bn.num_features}'\n"
        "\n"
        "# Shape behavior: (B, 1024, 4, 4) → (B, 512, 8, 8). Use eval() so BN is identity-ish.\n"
        "block.eval()\n"
        "with t.no_grad():\n"
        "    seed = t.randn(2, 1024, 4, 4)\n"
        "    out = block(seed)\n"
        "    assert out.shape == (2, 512, 8, 8), f'expected (2,512,8,8), got {tuple(out.shape)}'\n"
        "    # ReLU output → all nonnegative.\n"
        "    assert (out >= 0).all(), 'ReLU activation must zero out negatives'\n"
        "\n"
        "# A second build with different channels to confirm parametrization.\n"
        "small = ex1_build_generator_block(in_channels=128, out_channels=64)\n"
        "small.eval()\n"
        "with t.no_grad():\n"
        "    out_small = small(t.randn(1, 128, 16, 16))\n"
        "    assert out_small.shape == (1, 64, 32, 32), f'expected (1,64,32,32), got {tuple(out_small.shape)}'\n"
        "\n"
        "# --- Visualization: 4 output channels as 8x8 feature maps ---\n"
        "block.eval()\n"
        "with t.no_grad():\n"
        "    rng = t.Generator().manual_seed(0)\n"
        "    seed_viz = t.randn(1, 1024, 4, 4, generator=rng)\n"
        "    out_viz = block(seed_viz)\n"
        "fig, axes = plt.subplots(2, 2, figsize=(6, 6))\n"
        "for i, ax in enumerate(axes.flat):\n"
        "    ax.imshow(out_viz[0, i].numpy(), cmap='viridis')\n"
        "    ax.set_title(f'channel {i} (8×8)')\n"
        "    ax.axis('off')\n"
        "plt.suptitle('ex1 generator block output — 4 channel slices (4×4 → 8×8)')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_build_generator_block(in_channels: int, out_channels: int) -> t.nn.Sequential:\n"
        "    import torch.nn as nn\n"
        "    return nn.Sequential(\n"
        "        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),\n"
        "        nn.BatchNorm2d(out_channels),\n"
        "        nn.ReLU(inplace=True),\n"
        "    )"
    ),
    "solution_notes": (
        "**Why kernel=4, stride=2, padding=1.** This combination is the "
        "DCGAN paper standard. Output size for ConvTranspose2d is "
        "`H_out = (H_in - 1) * stride - 2 * padding + kernel = "
        "(H_in - 1) * 2 - 2 + 4 = 2 * H_in`. Exact doubling.\n\n"
        "**Why ReLU, not LeakyReLU.** Generator uses ReLU; discriminator "
        "uses LeakyReLU. The DCGAN authors found this asymmetry empirically "
        "stabilizes training. Generator needs crisp activations to "
        "produce sharp images; discriminator needs a gentle negative "
        "slope to keep learning from fakes.\n\n"
        "**`bias=False` matters.** Adding a bias to the ConvTranspose "
        "would be redundant since BatchNorm has its own `affine` "
        "parameters (`weight` and `bias`) and re-centres the entire "
        "feature map. Free parameter savings: `out_channels` per block."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
})


# -------------------------- ex1 / conv-leakyrelu-block-discriminator
SPECS.append({
    "atom_id": "conv-leakyrelu-block-discriminator",
    "subtopic": "GAN: Conv+LeakyReLU discriminator block",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_CONV_LR_DISC,
    "exercise_index": 1,
    "exercise_title": "build a 32x32 to 16x16 discriminator downsampling block",
    "slug": "conv-bn-leakyrelu-block",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["gan", "discriminator", "conv2d", "leakyrelu", "sequential"],
    "kcs": ["sequential-three-layer-block", "conv-stride-halves-spatial"],
    "lo": (
        "Apply `nn.Sequential` to wire `nn.Conv2d(stride=2, kernel=4, "
        "padding=1, bias=False) -> nn.BatchNorm2d -> nn.LeakyReLU(0.2)` "
        "into a single channel-doubling, spatial-halving discriminator block."
    ),
    "prompt_body": (
        "Implement `ex1_build_discriminator_block(in_channels, out_channels)`. "
        "The repeated unit of a DCGAN discriminator:\n\n"
        "1. Construct an `nn.Sequential` containing three layers IN ORDER:\n"
        "   - `nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, "
        "padding=1, bias=False)`\n"
        "   - `nn.BatchNorm2d(out_channels)`\n"
        "   - `nn.LeakyReLU(negative_slope=0.2, inplace=True)`\n"
        "2. `bias=False` on the Conv because BatchNorm immediately follows.\n"
        "3. Stride 2 + kernel 4 + padding 1 HALVES the spatial size — a "
        "32×32 input becomes 16×16 output.\n"
        "4. Return the Sequential.\n\n"
        "(Note: in a real DCGAN, the FIRST block omits BatchNorm because "
        "input is raw image stats. This drill builds an INTERMEDIATE block, "
        "so BN is included.)\n\n"
        "Input: `in_channels`, `out_channels` — ints.\n"
        "Output: `nn.Sequential` module.\n\n"
        "The visualization renders four output channel slices as a 2×2 "
        "grid of 16×16 feature maps."
    ),
    "stub": (
        "def ex1_build_discriminator_block(in_channels: int, out_channels: int) -> t.nn.Sequential:\n"
        '    """Return nn.Sequential(Conv2d, BatchNorm2d, LeakyReLU(0.2))."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "\n"
        "# Build a 32x32 -> 16x16 discriminator block (channel doubling).\n"
        "block = ex1_build_discriminator_block(in_channels=128, out_channels=256)\n"
        "assert isinstance(block, nn.Sequential), 'must return nn.Sequential'\n"
        "layers = list(block.children())\n"
        "assert len(layers) == 3, f'expected exactly 3 layers, got {len(layers)}'\n"
        "assert isinstance(layers[0], nn.Conv2d), f'layer 0 must be Conv2d, got {type(layers[0]).__name__}'\n"
        "assert isinstance(layers[1], nn.BatchNorm2d), f'layer 1 must be BatchNorm2d, got {type(layers[1]).__name__}'\n"
        "assert isinstance(layers[2], nn.LeakyReLU), f'layer 2 must be LeakyReLU, got {type(layers[2]).__name__}'\n"
        "\n"
        "# Conv configuration.\n"
        "cv = layers[0]\n"
        "assert cv.in_channels == 128 and cv.out_channels == 256, f'Conv channels wrong: in={cv.in_channels}, out={cv.out_channels}'\n"
        "assert cv.kernel_size == (4, 4), f'kernel must be 4, got {cv.kernel_size}'\n"
        "assert cv.stride == (2, 2), f'stride must be 2, got {cv.stride}'\n"
        "assert cv.padding == (1, 1), f'padding must be 1 for spatial-halving, got {cv.padding}'\n"
        "assert cv.bias is None, 'bias must be False (BatchNorm follows)'\n"
        "\n"
        "# BatchNorm configuration.\n"
        "bn = layers[1]\n"
        "assert bn.num_features == 256, f'BN num_features must match Conv out, got {bn.num_features}'\n"
        "\n"
        "# LeakyReLU slope.\n"
        "lr = layers[2]\n"
        "assert abs(lr.negative_slope - 0.2) < 1e-6, f'LeakyReLU slope must be 0.2, got {lr.negative_slope}'\n"
        "\n"
        "# Shape behavior: (B, 128, 32, 32) → (B, 256, 16, 16). Use eval() so BN is identity-ish.\n"
        "block.eval()\n"
        "with t.no_grad():\n"
        "    feats = t.randn(2, 128, 32, 32)\n"
        "    out = block(feats)\n"
        "    assert out.shape == (2, 256, 16, 16), f'expected (2,256,16,16), got {tuple(out.shape)}'\n"
        "    # LeakyReLU(0.2) — outputs can be negative, but negatives are SCALED by 0.2.\n"
        "    # The minimum should not be less than 0.2 * input_min - some bias.\n"
        "    # Simpler invariant: outputs must have BOTH positive and negative values for random input.\n"
        "    assert (out > 0).any() and (out < 0).any(), 'LeakyReLU output should have both signs'\n"
        "\n"
        "# A second build with different channels to confirm parametrization.\n"
        "small = ex1_build_discriminator_block(in_channels=3, out_channels=64)\n"
        "small.eval()\n"
        "with t.no_grad():\n"
        "    out_small = small(t.randn(1, 3, 64, 64))\n"
        "    assert out_small.shape == (1, 64, 32, 32), f'expected (1,64,32,32), got {tuple(out_small.shape)}'\n"
        "\n"
        "# --- Visualization: 4 output channels as 16x16 feature maps ---\n"
        "block.eval()\n"
        "with t.no_grad():\n"
        "    rng = t.Generator().manual_seed(0)\n"
        "    feats_viz = t.randn(1, 128, 32, 32, generator=rng)\n"
        "    out_viz = block(feats_viz)\n"
        "fig, axes = plt.subplots(2, 2, figsize=(6, 6))\n"
        "for i, ax in enumerate(axes.flat):\n"
        "    ax.imshow(out_viz[0, i].numpy(), cmap='magma')\n"
        "    ax.set_title(f'channel {i} (16×16)')\n"
        "    ax.axis('off')\n"
        "plt.suptitle('ex1 discriminator block output — 4 channel slices (32×32 → 16×16)')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_build_discriminator_block(in_channels: int, out_channels: int) -> t.nn.Sequential:\n"
        "    import torch.nn as nn\n"
        "    return nn.Sequential(\n"
        "        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),\n"
        "        nn.BatchNorm2d(out_channels),\n"
        "        nn.LeakyReLU(negative_slope=0.2, inplace=True),\n"
        "    )"
    ),
    "solution_notes": (
        "**Why kernel=4, stride=2, padding=1.** The discriminator mirror "
        "of the generator block. Output size for Conv2d is `H_out = "
        "(H_in + 2 * padding - kernel) // stride + 1 = (H_in - 2) // 2 + 1 "
        "= H_in // 2`. Exact halving.\n\n"
        "**Why LeakyReLU(0.2), not plain ReLU.** Plain ReLU zeroes out "
        "all negative inputs — including the discriminator's score for "
        "samples it currently thinks are fake. Gradient is zero on those, "
        "so the discriminator stops learning. LeakyReLU(0.2) gives a 0.2× "
        "gradient on negatives — still flowing, just attenuated. The 0.2 "
        "value comes straight from the DCGAN paper.\n\n"
        "**Real DCGAN: first block has NO BN.** The very first block "
        "downsamples raw image input (e.g. 64x64 RGB → 32x32 features). "
        "BatchNorm on raw pixel stats washes out the image. ARENA's "
        "reference implementation passes `skip_first_bn=True` to handle "
        "this — but here we build a clean intermediate block."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
})


# -------------------------- ex1 / discriminator-classifier-head
SPECS.append({
    "atom_id": "discriminator-classifier-head",
    "subtopic": "GAN: Discriminator classifier head",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_DISC_HEAD,
    "exercise_index": 1,
    "exercise_title": "flatten + linear + sigmoid scalar real/fake head",
    "slug": "flatten-linear-sigmoid-head",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["gan", "discriminator", "sigmoid", "flatten", "head"],
    "kcs": ["flatten-from-axis-1", "sigmoid-binary-prob"],
    "lo": (
        "Apply `flatten(start_dim=1) -> Linear(features, 1) -> sigmoid -> "
        "squeeze(-1)` to map a `(B, C, H, W)` feature map to a `(B,)` "
        "real-vs-fake probability vector."
    ),
    "prompt_body": (
        "Implement `ex1_discriminator_head(features, weight, bias)`. The "
        "scalar real/fake probability head:\n\n"
        "1. `features` has shape `(B, C, H, W)` — output of the last "
        "discriminator block.\n"
        "2. `weight` has shape `(1, C * H * W)` and `bias` has shape "
        "`(1,)` — together they parameterize `nn.Linear(C * H * W, 1)`.\n"
        "3. Flatten everything except the batch axis: "
        "`flat = features.flatten(start_dim=1)` → `(B, C * H * W)`.\n"
        "4. Affine to scalar: `logits = flat @ weight.T + bias` → `(B, 1)`.\n"
        "5. Sigmoid then squeeze the trailing axis: "
        "`probs = t.sigmoid(logits).squeeze(-1)` → `(B,)`.\n"
        "6. Return the `(B,)` probability vector — all values in `(0, 1)`.\n\n"
        "Input: `features` `(B, C, H, W)`; `weight` `(1, C*H*W)`; `bias` "
        "`(1,)`.\n"
        "Output: `(B,)` float tensor with values in `(0, 1)`.\n\n"
        "The visualization plots the histogram of probabilities the head "
        "produces on a synthetic mixture of 'real-leaning' and 'fake-"
        "leaning' feature maps."
    ),
    "stub": (
        "def ex1_discriminator_head(features: Tensor, weight: Tensor, bias: Tensor) -> Tensor:\n"
        '    """Flatten + linear + sigmoid → (B,) real-prob."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Canonical DCGAN shape: (B, 1024, 4, 4) → (B,).\n"
        "B, C, H = 4, 1024, 4\n"
        "rng = t.Generator().manual_seed(0)\n"
        "feats = t.randn(B, C, H, H, generator=rng)\n"
        "W = 0.01 * t.randn(1, C * H * H, generator=rng)\n"
        "b = t.zeros(1)\n"
        "probs = ex1_discriminator_head(feats, W, b)\n"
        "assert probs.shape == (B,), f'expected (B,), got {tuple(probs.shape)}'\n"
        "assert probs.dtype == t.float32\n"
        "# All probabilities must be in (0, 1) (sigmoid range).\n"
        "assert (probs > 0).all() and (probs < 1).all(), f'sigmoid output must be in (0, 1), got {probs}'\n"
        "\n"
        "# Numerical correctness vs manual computation.\n"
        "flat_ref = feats.flatten(start_dim=1)\n"
        "logits_ref = flat_ref @ W.T + b\n"
        "expected = t.sigmoid(logits_ref).squeeze(-1)\n"
        "assert t.allclose(probs, expected, atol=1e-5), 'output must equal sigmoid(flatten @ W.T + b).squeeze(-1)'\n"
        "\n"
        "# Bias-only behavior: features=0, then probs depend only on bias.\n"
        "zero_feats = t.zeros(3, C, H, H)\n"
        "# bias=0 → sigmoid(0) = 0.5.\n"
        "probs_b0 = ex1_discriminator_head(zero_feats, W, t.zeros(1))\n"
        "assert t.allclose(probs_b0, t.full((3,), 0.5), atol=1e-5), f'b=0 sigmoid output must be 0.5, got {probs_b0}'\n"
        "# bias=very large positive → probs ≈ 1.\n"
        "probs_bpos = ex1_discriminator_head(zero_feats, W, t.tensor([10.0]))\n"
        "assert (probs_bpos > 0.99).all(), f'large positive bias → probs near 1, got {probs_bpos}'\n"
        "# bias=very large negative → probs ≈ 0.\n"
        "probs_bneg = ex1_discriminator_head(zero_feats, W, t.tensor([-10.0]))\n"
        "assert (probs_bneg < 0.01).all(), f'large negative bias → probs near 0, got {probs_bneg}'\n"
        "\n"
        "# Smaller shape — make sure flatten generalizes.\n"
        "feats_small = t.randn(2, 8, 2, 2)\n"
        "W_small = t.randn(1, 8 * 2 * 2)\n"
        "b_small = t.zeros(1)\n"
        "probs_small = ex1_discriminator_head(feats_small, W_small, b_small)\n"
        "assert probs_small.shape == (2,)\n"
        "\n"
        "# --- Visualization: histogram on 'real' vs 'fake' synthetic feature maps ---\n"
        "rng = t.Generator().manual_seed(11)\n"
        "B_viz = 500\n"
        "C_viz, H_viz = 32, 4\n"
        "# Half the batch has positive-mean features (real-leaning), half negative-mean (fake-leaning).\n"
        "real_like = 0.5 + 0.3 * t.randn(B_viz // 2, C_viz, H_viz, H_viz, generator=rng)\n"
        "fake_like = -0.5 + 0.3 * t.randn(B_viz // 2, C_viz, H_viz, H_viz, generator=rng)\n"
        "all_feats = t.cat([real_like, fake_like], dim=0)\n"
        "# A weight that's positive everywhere → real-mean features get higher logit.\n"
        "W_viz = (1.0 / (C_viz * H_viz * H_viz)) * t.ones(1, C_viz * H_viz * H_viz)\n"
        "b_viz = t.zeros(1)\n"
        "all_probs = ex1_discriminator_head(all_feats, W_viz, b_viz)\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "ax.hist(all_probs[:B_viz // 2].numpy(), bins=30, alpha=0.6, label='real-leaning input', color='steelblue', edgecolor='black')\n"
        "ax.hist(all_probs[B_viz // 2:].numpy(), bins=30, alpha=0.6, label='fake-leaning input', color='crimson', edgecolor='black')\n"
        "ax.axvline(0.5, color='black', ls='--', label='P=0.5 decision boundary')\n"
        "ax.set_xlabel('discriminator P(real)')\n"
        "ax.set_ylabel('count')\n"
        "ax.set_title('ex1 discriminator head — probability histogram')\n"
        "ax.legend()\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_discriminator_head(features: Tensor, weight: Tensor, bias: Tensor) -> Tensor:\n"
        "    flat = features.flatten(start_dim=1)             # (B, C*H*W)\n"
        "    logits = flat @ weight.T + bias                  # (B, 1)\n"
        "    return t.sigmoid(logits).squeeze(-1)             # (B,)"
    ),
    "solution_notes": (
        "**Why `flatten(start_dim=1)`.** Collapses the channel + spatial "
        "axes into one feature axis while preserving the batch axis. "
        "`features.view(B, -1)` does the same thing more cryptically. "
        "Either works on contiguous tensors.\n\n"
        "**Sigmoid, not softmax.** The output is binary (real vs fake) "
        "with ONE positive class. `P(fake) = 1 - P(real)` is implicit. "
        "Softmax over a single output is degenerate (always 1).\n\n"
        "**Returning probs vs logits.** In a real training loop you'd "
        "return LOGITS and pair with `F.binary_cross_entropy_with_logits` "
        "for numerical stability (fuses sigmoid with BCE, avoids `log(0)` "
        "when sigmoid saturates). The explicit `sigmoid` here is for "
        "INFERENCE — to interpret the output as a probability — and to "
        "match the per-step inspection ARENA uses while debugging.\n\n"
        "**`.squeeze(-1)` matters.** Without it the output is `(B, 1)`, "
        "which broadcasts against `(B,)` labels in confusing ways. The "
        "squeeze pins the head to `(B,)` so downstream code is unambiguous."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
})


# ---------------------------------------------------------------- emit

for spec in SPECS:
    path = emit_standalone(spec)
    rel = path.relative_to(path.parents[3])
    print(f"wrote {rel}")
