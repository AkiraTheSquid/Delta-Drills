#!/usr/bin/env python3
"""Author Colab-native standalones for ARENA chapter-0 part-5 (DCGAN/GAN) final atoms.

Batch 7: eight single-exercise drills for the remaining DCGAN/GAN component atoms
under `prereqs_dcgan_final/`. Each atom gets ONE ex (ex1). Each exercise hits
ONE Bloom level + at most 2 KCs.

Atom roster (8):
    - dcgan-normal-init-002              (nn.init.normal_(weight, 0.0, 0.02) on Conv/ConvT via apply)
    - bn-weight-bias-init-pattern        (BatchNorm: weight ~ N(1, 0.02), bias = 0)
    - module-modules-iter-isinstance-dispatch (for m in model.modules(): isinstance dispatch)
    - channel-list-reverse-build         (decoder = encoder channels reversed)
    - bce-log-loss-real-fake             (D loss = BCE(D(real), 1) + BCE(D(fake), 0))
    - generator-loss-fool-discriminator  (G loss = BCE(D(G(z)), 1))
    - noise-batch-from-latent            (t.randn(B, latent_dim, 1, 1) spatial-prefix noise)
    - model-train-eval-toggle-around-sample (eval -> no_grad -> sample -> train)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_dcgan_final"


# ---------------------------------------------------------------- recaps

RECAP_NORMAL_INIT = (
    "## DCGAN normal init (mean=0, std=0.02) — quick refresher\n"
    "\n"
    "The Radford et al. DCGAN paper specifies a fixed init for every Conv / "
    "ConvTranspose weight: `N(0, 0.02)`. The canonical implementation walks "
    "the model and applies the init to matching layers:\n"
    "\n"
    "```python\n"
    "def init_dcgan(m):\n"
    "    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):\n"
    "        nn.init.normal_(m.weight, mean=0.0, std=0.02)\n"
    "\n"
    "model.apply(init_dcgan)\n"
    "```\n"
    "\n"
    "**Why std=0.02, not the PyTorch default.** The default for `Conv2d` is "
    "Kaiming-uniform — fine for ReLU classifiers but too wide for adversarial "
    "training. A small std (0.02) keeps activations bounded early in training "
    "so the discriminator doesn't immediately saturate and starve the "
    "generator of gradient.\n"
    "\n"
    "**`model.apply(fn)` recurses.** Walks every submodule (depth-first) and "
    "calls `fn(submodule)`. The function should be a no-op for layers it "
    "doesn't recognize — hence the `isinstance` guard."
)

RECAP_BN_INIT = (
    "## BatchNorm weight=N(1, 0.02), bias=0 — quick refresher\n"
    "\n"
    "BatchNorm has its own DCGAN init: weight (gamma) sampled from `N(1.0, "
    "0.02)`, bias (beta) set to zero.\n"
    "\n"
    "```python\n"
    "if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):\n"
    "    nn.init.normal_(m.weight, 1.0, 0.02)\n"
    "    nn.init.zeros_(m.bias)\n"
    "```\n"
    "\n"
    "**Why mean=1 for weight.** BatchNorm's affine transform is `gamma * "
    "normalized + beta`. At initialization we want the layer to be near "
    "identity (just pass the normalized features through), so `gamma ≈ 1` "
    "and `beta = 0`. PyTorch's default already does this — DCGAN adds the "
    "small jitter (`std=0.02`) to break symmetry without disturbing scale.\n"
    "\n"
    "**`nn.init.zeros_(m.bias)` vs `m.bias.data.zero_()`.** Both work. The "
    "`nn.init` helpers are the convention in modern PyTorch code — they "
    "respect the no-grad context and read more clearly."
)

RECAP_MODULES_DISPATCH = (
    "## model.modules() + isinstance dispatch — quick refresher\n"
    "\n"
    "`model.modules()` is an iterator over EVERY submodule in the network "
    "(including the model itself, recursively). Pair it with `isinstance` to "
    "do per-layer-type work without `apply`:\n"
    "\n"
    "```python\n"
    "for m in model.modules():\n"
    "    if isinstance(m, nn.Conv2d):\n"
    "        nn.init.normal_(m.weight, 0.0, 0.02)\n"
    "    elif isinstance(m, nn.BatchNorm2d):\n"
    "        nn.init.normal_(m.weight, 1.0, 0.02)\n"
    "        nn.init.zeros_(m.bias)\n"
    "```\n"
    "\n"
    "**`modules()` vs `apply(fn)`.** `apply` calls `fn(m)` on every "
    "submodule too — but `modules()` gives you a plain Python loop where "
    "you can branch, count, accumulate stats, or break early. Use `apply` "
    "for pure transforms; use `modules()` when you want a procedural body.\n"
    "\n"
    "**`modules()` vs `children()`.** `children()` is shallow — only the "
    "DIRECT submodules. `modules()` is deep — every descendant. For init "
    "you almost always want `modules()`."
)

RECAP_CHANNEL_REVERSE = (
    "## Channel-list reverse build (encoder → decoder) — quick refresher\n"
    "\n"
    "A symmetric autoencoder / DCGAN-style architecture uses the encoder's "
    "channel list, REVERSED, for the decoder. The convention reads "
    "naturally — the decoder unrolls the encoder.\n"
    "\n"
    "```python\n"
    "encoder_channels = [3, 64, 128, 256, 512]       # input → bottleneck\n"
    "decoder_channels = encoder_channels[::-1]       # [512, 256, 128, 64, 3]\n"
    "```\n"
    "\n"
    "Then you walk consecutive pairs to build blocks:\n"
    "```python\n"
    "for in_c, out_c in zip(decoder_channels[:-1], decoder_channels[1:]):\n"
    "    blocks.append(convt_block(in_c, out_c))\n"
    "```\n"
    "\n"
    "**Slice `[::-1]` vs `list(reversed(x))`.** Both work for a Python "
    "list. The slice form is one token shorter and emphasizes the "
    "structural symmetry — encoder is `c`, decoder is `c[::-1]`. ARENA "
    "uses the slice form.\n"
    "\n"
    "**Why this matters.** Hardcoding two parallel lists is a recipe for "
    "drift: change the encoder, forget the decoder, mysterious shape "
    "mismatch three commits later. Deriving one from the other guarantees "
    "they stay in sync."
)

RECAP_BCE_REAL_FAKE = (
    "## BCE loss for D (real + fake) — quick refresher\n"
    "\n"
    "The discriminator is a binary classifier — its loss is the SUM of two "
    "BCE terms, one for real images (target 1), one for fakes (target 0):\n"
    "\n"
    "```python\n"
    "loss_real = F.binary_cross_entropy(D(reals), t.ones_like(D(reals)))\n"
    "loss_fake = F.binary_cross_entropy(D(fakes), t.zeros_like(D(fakes)))\n"
    "loss_D = loss_real + loss_fake\n"
    "```\n"
    "\n"
    "**Sum, not mean across the two terms.** Each BCE is already MEAN-"
    "reduced across the batch. Adding the two gives the discriminator twice "
    "as many gradient steps per batch effectively — the standard recipe.\n"
    "\n"
    "**`ones_like` / `zeros_like` instead of constants.** Matches shape "
    "dtype device automatically; `t.ones(batch_size)` silently breaks if "
    "D outputs `(B, 1)` not `(B,)`.\n"
    "\n"
    "**In practice, use `binary_cross_entropy_with_logits`.** The version "
    "that fuses sigmoid + BCE is numerically stable. The bare `F.binary_"
    "cross_entropy` (this recap) expects probabilities in `[0, 1]` — feed "
    "it sigmoid outputs, not logits."
)

RECAP_G_LOSS = (
    "## Generator loss (fool the discriminator) — quick refresher\n"
    "\n"
    "The generator's loss is BCE on D's verdict for the fake batch, with "
    "target 1 — i.e. the generator WANTS D to call its fakes real:\n"
    "\n"
    "```python\n"
    "fakes = G(noise)\n"
    "d_pred = D(fakes)\n"
    "loss_G = F.binary_cross_entropy(d_pred, t.ones_like(d_pred))\n"
    "```\n"
    "\n"
    "**Note the asymmetry vs the D loss.** When D trains, the fake target "
    "is 0. When G trains, the fake target is 1 — same fakes, opposite "
    "label. That's the adversarial part.\n"
    "\n"
    "**Why not `-loss_D_fake` (the math-paper form).** Goodfellow's "
    "original `min_G max_D` formulation gives `G_loss = log(1 - D(G(z)))` "
    "— but that gradient vanishes when D is confident the fake is fake "
    "(which is most of training). The flipped-target BCE — `-log(D(G(z)))` "
    "— gives strong gradient exactly when G needs it. This is the 'non-"
    "saturating loss' from the paper.\n"
    "\n"
    "**Keep D frozen here.** Only G's parameters get gradient on this loss "
    "in the standard GAN training loop. The framework handles that via the "
    "optimizer split — `optim_G.step()` only touches G's params."
)

RECAP_NOISE_BATCH = (
    "## DCGAN noise batch (spatial latent) — quick refresher\n"
    "\n"
    "The DCGAN generator's input is a 4-D noise tensor, not a flat vector — "
    "the spatial dims are 1×1 placeholders that ConvTranspose layers will "
    "expand:\n"
    "\n"
    "```python\n"
    "noise = t.randn(batch_size, latent_dim, 1, 1, device=device)\n"
    "fakes = G(noise)   # G is built of ConvTranspose2d layers\n"
    "```\n"
    "\n"
    "Shape contract: `(B, latent_dim, 1, 1)`. The trailing 1×1 makes it a "
    "valid input to `ConvTranspose2d(latent_dim, ..., kernel_size=4)` — "
    "the first transposed conv blows the spatial dims out to 4×4.\n"
    "\n"
    "**Compared to flat `(B, latent_dim) + view + Linear` (the ARENA "
    "form).** Some implementations use `Linear` then `view` to a spatial "
    "seed; others go straight from `(B, latent_dim, 1, 1)` through "
    "`ConvTranspose`. Both produce identical output shapes; the "
    "fully-convolutional form (this recap) is the original DCGAN paper.\n"
    "\n"
    "**Standard-normal noise, not uniform.** `t.randn` (not `t.rand`) — "
    "the prior is `N(0, 1)`. Uniform-noise priors exist but are not the "
    "DCGAN default and produce different visual artifacts."
)

RECAP_TRAIN_EVAL_TOGGLE = (
    "## model.train() / eval() toggle around sample — quick refresher\n"
    "\n"
    "When you sample from a generator (or any net with BatchNorm / Dropout) "
    "for LOGGING or VISUALIZATION inside a training loop, you must:\n"
    "\n"
    "```python\n"
    "model.eval()                              # turn BN to running stats, disable dropout\n"
    "with t.no_grad():                         # don't track this in autograd\n"
    "    samples = model(noise)\n"
    "model.train()                             # restore training mode\n"
    "```\n"
    "\n"
    "**Why both `eval()` and `no_grad()`.** They do different things. "
    "`eval()` switches the BEHAVIOR of BatchNorm (use running stats, don't "
    "update them) and Dropout (no zeroing). `no_grad()` switches the "
    "GRADIENT MACHINERY (no graph, no `requires_grad` propagation). You "
    "need both for a clean sample.\n"
    "\n"
    "**Without `eval()`.** BatchNorm computes mean/var on the noise batch "
    "(which is unrelated to your real-data running stats) AND updates "
    "those running stats — corrupting the model's BN parameters with "
    "noise. Visible later as quality regression when training resumes.\n"
    "\n"
    "**Why restore with `model.train()`.** The next training step needs "
    "the model in training mode. Forgetting this is one of the classic "
    "GAN bugs — BatchNorm running stats freeze, gradients flow weirdly, "
    "your loss curve mysteriously plateaus."
)


# ---------------------------------------------------------------- specs

SPECS = []

# -------------------------- ex1 / dcgan-normal-init-002
SPECS.append({
    "atom_id": "dcgan-normal-init-002",
    "subtopic": "GAN: DCGAN normal init 0.02",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_NORMAL_INIT,
    "exercise_index": 1,
    "exercise_title": "apply normal(0, 0.02) init to Conv/ConvTranspose via model.apply",
    "slug": "apply-dcgan-normal-init-via-apply",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["dcgan", "init", "model.apply", "normal-init"],
    "kcs": ["dcgan-conv-normal-init", "model-apply-recursion"],
    "lo": (
        "Apply `nn.init.normal_(m.weight, 0.0, 0.02)` to every Conv2d and "
        "ConvTranspose2d submodule of a model using `model.apply(init_fn)`, "
        "leaving other layer types untouched."
    ),
    "prompt_body": (
        "Implement `ex1_apply_dcgan_init(model)`. The Radford et al. DCGAN "
        "convolutional weight initializer:\n\n"
        "1. Define a local function `init_fn(m)` that:\n"
        "   - Checks `isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))`.\n"
        "   - If so, calls `nn.init.normal_(m.weight, mean=0.0, std=0.02)` "
        "(in-place).\n"
        "   - Does NOTHING for other module types (BatchNorm, Linear, "
        "Sequential, the model itself).\n"
        "2. Call `model.apply(init_fn)` to walk every submodule.\n"
        "3. Return `model` (mutated in place, but returning is conventional).\n\n"
        "Input: `model` — `nn.Module`, may contain Conv2d, ConvTranspose2d, "
        "BatchNorm2d, Linear submodules (mix any).\n"
        "Output: same `model` with Conv / ConvT weights resampled from "
        "`N(0, 0.02)`. Other layers untouched.\n\n"
        "The visualization runs your init on a small DCGAN-style model and "
        "renders before/after histograms of Conv2d weight values so you can "
        "verify the distribution change."
    ),
    "stub": (
        "def ex1_apply_dcgan_init(model: nn.Module) -> nn.Module:\n"
        '    """Apply N(0, 0.02) to Conv/ConvTranspose weights via model.apply."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "\n"
        "# Build a mixed model — has Conv, ConvT, BN, Linear.\n"
        "model = nn.Sequential(\n"
        "    nn.Conv2d(3, 16, 3, padding=1),\n"
        "    nn.BatchNorm2d(16),\n"
        "    nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),\n"
        "    nn.Conv2d(8, 4, 1),\n"
        "    nn.Flatten(),\n"
        "    nn.Linear(64, 10),\n"
        ")\n"
        "\n"
        "# Snapshot BN gamma + Linear weight BEFORE init — must remain unchanged.\n"
        "bn_w_before = model[1].weight.detach().clone()\n"
        "lin_w_before = model[5].weight.detach().clone()\n"
        "\n"
        "out = ex1_apply_dcgan_init(model)\n"
        "assert out is model, 'must return the same model (mutated in place)'\n"
        "\n"
        "# Conv layers must now have weight std ~ 0.02 (sample-size loose tolerance).\n"
        "conv_layers = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]\n"
        "assert len(conv_layers) == 3, f'expected 3 conv layers, found {len(conv_layers)}'\n"
        "for layer in conv_layers:\n"
        "    s = layer.weight.std().item()\n"
        "    mean = layer.weight.mean().item()\n"
        "    assert abs(s - 0.02) < 0.01, f'expected std ~0.02, got {s:.5f} for {type(layer).__name__}'\n"
        "    assert abs(mean) < 0.01, f'expected mean ~0, got {mean:.5f}'\n"
        "\n"
        "# BatchNorm weight + Linear weight must be UNCHANGED.\n"
        "assert t.equal(model[1].weight, bn_w_before), 'BatchNorm weight must be untouched'\n"
        "assert t.equal(model[5].weight, lin_w_before), 'Linear weight must be untouched'\n"
        "\n"
        "# Stress test — nested Sequential, still walks recursively.\n"
        "nested = nn.Sequential(\n"
        "    nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8)),\n"
        "    nn.Sequential(nn.ConvTranspose2d(8, 16, 4)),\n"
        ")\n"
        "ex1_apply_dcgan_init(nested)\n"
        "for m in nested.modules():\n"
        "    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):\n"
        "        assert abs(m.weight.std().item() - 0.02) < 0.015, 'nested conv weight not initialized'\n"
        "\n"
        "# --- Visualization: weight histogram before vs after on a fresh model ---\n"
        "viz_model = nn.Sequential(\n"
        "    nn.Conv2d(3, 64, 4, stride=2, padding=1),\n"
        "    nn.Conv2d(64, 128, 4, stride=2, padding=1),\n"
        "    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),\n"
        ")\n"
        "before_vals = t.cat([m.weight.detach().flatten() for m in viz_model.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))])\n"
        "ex1_apply_dcgan_init(viz_model)\n"
        "after_vals = t.cat([m.weight.detach().flatten() for m in viz_model.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))])\n"
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)\n"
        "ax1.hist(before_vals.numpy(), bins=60, color='gray', edgecolor='black')\n"
        "ax1.set_title(f'before init — std={before_vals.std().item():.4f}')\n"
        "ax1.set_xlabel('weight value'); ax1.set_ylabel('count')\n"
        "ax2.hist(after_vals.numpy(), bins=60, color='steelblue', edgecolor='black')\n"
        "ax2.set_title(f'after DCGAN init — std={after_vals.std().item():.4f}')\n"
        "ax2.set_xlabel('weight value')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_apply_dcgan_init(model: nn.Module) -> nn.Module:\n"
        "    def init_fn(m):\n"
        "        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):\n"
        "            nn.init.normal_(m.weight, mean=0.0, std=0.02)\n"
        "    model.apply(init_fn)\n"
        "    return model"
    ),
    "solution_notes": (
        "**`model.apply` does the recursion for you.** No need to write `for "
        "m in model.modules()` — `apply` walks every submodule and calls "
        "`init_fn(m)`. The `isinstance` guard inside keeps the function a "
        "no-op for layers we don't want to touch.\n\n"
        "**Why std=0.02 specifically.** Radford et al. picked this by "
        "experiment — small enough to prevent immediate D saturation, large "
        "enough to break symmetry. Don't substitute Kaiming / Xavier in a "
        "DCGAN; the gradient balance breaks.\n\n"
        "**Bias is untouched.** DCGAN convolutions often have `bias=False` "
        "(because BatchNorm follows). When `bias=True`, leaving it at "
        "PyTorch's default zero-init is fine."
    ),
    "extra_imports": [
        "import torch.nn as nn",
        "import matplotlib.pyplot as plt",
    ],
})


# -------------------------- ex1 / bn-weight-bias-init-pattern
SPECS.append({
    "atom_id": "bn-weight-bias-init-pattern",
    "subtopic": "GAN: BN weight=1 bias=0 init",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_BN_INIT,
    "exercise_index": 1,
    "exercise_title": "BatchNorm init — weight ~ N(1, 0.02), bias = 0",
    "slug": "bn-weight-normal1-bias-zero-init",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["dcgan", "batchnorm", "init", "gamma-beta"],
    "kcs": ["bn-weight-normal-mean-1", "bn-bias-zero"],
    "lo": (
        "Apply `nn.init.normal_(m.weight, 1.0, 0.02)` and `nn.init.zeros_(m."
        "bias)` to every BatchNorm submodule of a model, leaving non-BN "
        "layers untouched."
    ),
    "prompt_body": (
        "Implement `ex1_apply_bn_init(model)`. The DCGAN BatchNorm "
        "initialization:\n\n"
        "1. Define `init_fn(m)` that:\n"
        "   - Checks `isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))`.\n"
        "   - If so, calls `nn.init.normal_(m.weight, 1.0, 0.02)` and "
        "`nn.init.zeros_(m.bias)`.\n"
        "   - Does NOTHING for other module types.\n"
        "2. Call `model.apply(init_fn)`.\n"
        "3. Return `model`.\n\n"
        "Input: `model` — `nn.Module`, may contain BatchNorm + other layer "
        "types (Conv, Linear, etc.).\n"
        "Output: same `model` with BN gamma resampled from `N(1, 0.02)` and "
        "BN beta zeroed.\n\n"
        "The visualization runs your init on a model and renders BN weight "
        "(gamma) and bias (beta) before/after as two pairs of histograms."
    ),
    "stub": (
        "def ex1_apply_bn_init(model: nn.Module) -> nn.Module:\n"
        '    """Init BatchNorm weight ~ N(1, 0.02), bias = 0."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "\n"
        "# Build a mixed model — has BN1d, BN2d, Conv, Linear.\n"
        "model = nn.Sequential(\n"
        "    nn.Conv2d(3, 16, 3, padding=1),\n"
        "    nn.BatchNorm2d(16),\n"
        "    nn.Conv2d(16, 32, 3, padding=1),\n"
        "    nn.BatchNorm2d(32),\n"
        "    nn.Flatten(),\n"
        "    nn.Linear(32 * 4, 64),\n"
        "    nn.BatchNorm1d(64),\n"
        ")\n"
        "\n"
        "# Snapshot Conv + Linear weights BEFORE — they must stay unchanged.\n"
        "conv_w_before = model[0].weight.detach().clone()\n"
        "lin_w_before = model[5].weight.detach().clone()\n"
        "\n"
        "out = ex1_apply_bn_init(model)\n"
        "assert out is model, 'must return the same model'\n"
        "\n"
        "# BN layers must have weight near 1, bias exactly 0.\n"
        "bn_layers = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]\n"
        "assert len(bn_layers) == 3, f'expected 3 BN layers, got {len(bn_layers)}'\n"
        "for bn in bn_layers:\n"
        "    w_mean = bn.weight.mean().item()\n"
        "    w_std = bn.weight.std().item()\n"
        "    assert abs(w_mean - 1.0) < 0.05, f'BN weight mean expected ~1, got {w_mean:.4f}'\n"
        "    assert abs(w_std - 0.02) < 0.02, f'BN weight std expected ~0.02, got {w_std:.5f}'\n"
        "    assert t.all(bn.bias == 0), f'BN bias must be exactly zero, got mean {bn.bias.mean().item()}'\n"
        "\n"
        "# Conv + Linear must be UNCHANGED.\n"
        "assert t.equal(model[0].weight, conv_w_before), 'Conv weight must be untouched'\n"
        "assert t.equal(model[5].weight, lin_w_before), 'Linear weight must be untouched'\n"
        "\n"
        "# --- Visualization: BN gamma + beta before/after ---\n"
        "viz = nn.Sequential(\n"
        "    nn.BatchNorm2d(128),\n"
        "    nn.BatchNorm2d(256),\n"
        "    nn.BatchNorm1d(512),\n"
        ")\n"
        "w_before = t.cat([m.weight.detach().flatten() for m in viz.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))])\n"
        "b_before = t.cat([m.bias.detach().flatten() for m in viz.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))])\n"
        "# Perturb bias so we can see the zero-out happen.\n"
        "for m in viz.modules():\n"
        "    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):\n"
        "        with t.no_grad(): m.bias.add_(0.5)\n"
        "ex1_apply_bn_init(viz)\n"
        "w_after = t.cat([m.weight.detach().flatten() for m in viz.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))])\n"
        "b_after = t.cat([m.bias.detach().flatten() for m in viz.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))])\n"
        "fig, axes = plt.subplots(2, 2, figsize=(10, 6))\n"
        "axes[0, 0].hist(w_before.numpy(), bins=40, color='gray', edgecolor='black')\n"
        "axes[0, 0].set_title(f'weight before — mean={w_before.mean().item():.3f}, std={w_before.std().item():.3f}')\n"
        "axes[0, 1].hist(w_after.numpy(), bins=40, color='steelblue', edgecolor='black')\n"
        "axes[0, 1].set_title(f'weight after — mean={w_after.mean().item():.3f}, std={w_after.std().item():.3f}')\n"
        "axes[1, 0].hist(b_before.numpy(), bins=40, color='gray', edgecolor='black')\n"
        "axes[1, 0].set_title(f'bias before (perturbed) — mean={b_before.mean().item():.3f}')\n"
        "axes[1, 1].hist(b_after.numpy(), bins=40, color='coral', edgecolor='black')\n"
        "axes[1, 1].set_title(f'bias after — mean={b_after.mean().item():.3f}')\n"
        "for ax in axes.flat:\n"
        "    ax.set_xlabel('value')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_apply_bn_init(model: nn.Module) -> nn.Module:\n"
        "    def init_fn(m):\n"
        "        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):\n"
        "            nn.init.normal_(m.weight, 1.0, 0.02)\n"
        "            nn.init.zeros_(m.bias)\n"
        "    model.apply(init_fn)\n"
        "    return model"
    ),
    "solution_notes": (
        "**Why N(1, 0.02), not just `ones_`.** PyTorch's default is already "
        "gamma=1, beta=0 — the DCGAN convention adds a tiny random jitter on "
        "gamma to break perfect symmetry across channels. Helps the "
        "discriminator find different patterns per channel early in training.\n\n"
        "**Catch both BN1d and BN2d.** BatchNorm1d shows up on Linear "
        "outputs in the discriminator's classifier head; BatchNorm2d shows "
        "up between Conv/ConvT layers. The same init applies to both — your "
        "`isinstance` tuple must include both.\n\n"
        "**`nn.init.zeros_(m.bias)` is in-place.** Like all `nn.init.*_` "
        "helpers, it modifies the tensor and returns it; no need to assign "
        "the result. Same with `normal_`, `uniform_`, `kaiming_uniform_`, "
        "etc."
    ),
    "extra_imports": [
        "import torch.nn as nn",
        "import matplotlib.pyplot as plt",
    ],
})


# -------------------------- ex1 / module-modules-iter-isinstance-dispatch
SPECS.append({
    "atom_id": "module-modules-iter-isinstance-dispatch",
    "subtopic": "GAN: model.modules() isinstance dispatch",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_MODULES_DISPATCH,
    "exercise_index": 1,
    "exercise_title": "count + tag layers by type using model.modules() + isinstance",
    "slug": "modules-iter-isinstance-count",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["modules", "isinstance", "dispatch", "inspection"],
    "kcs": ["modules-iter-recursive", "isinstance-layer-dispatch"],
    "lo": (
        "Apply `model.modules()` plus `isinstance` dispatch to walk every "
        "submodule of a network and accumulate a count per layer type into a "
        "dictionary."
    ),
    "prompt_body": (
        "Implement `ex1_count_layer_types(model)`. The procedural-loop "
        "version of `model.apply` — useful when you need branch + branch + "
        "accumulator, not a pure transform:\n\n"
        "1. Initialize a dict `counts` with keys `'conv2d'`, "
        "`'convtranspose2d'`, `'batchnorm'`, `'linear'`, `'other'`, all set "
        "to 0.\n"
        "2. Iterate `for m in model.modules()`. (`modules()` is recursive — "
        "it yields the model itself plus every nested submodule.)\n"
        "3. Dispatch by `isinstance`:\n"
        "   - `nn.Conv2d` → bump `'conv2d'`\n"
        "   - `nn.ConvTranspose2d` → bump `'convtranspose2d'`\n"
        "   - `nn.BatchNorm1d` or `nn.BatchNorm2d` → bump `'batchnorm'`\n"
        "   - `nn.Linear` → bump `'linear'`\n"
        "   - any other type → bump `'other'`\n"
        "4. Return `counts`.\n\n"
        "Important: each module gets counted EXACTLY ONCE. The branches must "
        "be mutually exclusive — use `if / elif`, not stacked `if`s.\n\n"
        "Input: `model` — `nn.Module`.\n"
        "Output: dict[str, int] with the five keys above.\n\n"
        "The visualization runs your counter on a DCGAN-shaped model and "
        "renders the layer-type counts as a horizontal bar chart."
    ),
    "stub": (
        "def ex1_count_layer_types(model: nn.Module) -> dict:\n"
        '    """Count submodules by type via model.modules() + isinstance dispatch."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "\n"
        "# Hand-built model with known counts.\n"
        "model = nn.Sequential(\n"
        "    nn.Conv2d(3, 32, 3, padding=1),     # conv2d\n"
        "    nn.BatchNorm2d(32),                  # batchnorm\n"
        "    nn.ReLU(),                           # other\n"
        "    nn.Conv2d(32, 64, 3, padding=1),    # conv2d\n"
        "    nn.BatchNorm2d(64),                  # batchnorm\n"
        "    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # convtranspose2d\n"
        "    nn.Flatten(),                        # other\n"
        "    nn.Linear(32 * 4 * 4, 10),           # linear\n"
        ")\n"
        "counts = ex1_count_layer_types(model)\n"
        "assert isinstance(counts, dict), f'expected dict, got {type(counts).__name__}'\n"
        "expected_keys = {'conv2d', 'convtranspose2d', 'batchnorm', 'linear', 'other'}\n"
        "assert set(counts.keys()) == expected_keys, f'keys wrong: {set(counts.keys())}'\n"
        "assert counts['conv2d'] == 2, f'conv2d count wrong: {counts}'\n"
        "assert counts['convtranspose2d'] == 1, f'convtranspose2d count wrong: {counts}'\n"
        "assert counts['batchnorm'] == 2, f'batchnorm count wrong: {counts}'\n"
        "assert counts['linear'] == 1, f'linear count wrong: {counts}'\n"
        "# 'other' includes ReLU + Flatten + Sequential (the model itself).\n"
        "assert counts['other'] == 3, f'other count wrong: {counts}'\n"
        "\n"
        "# Empty model — only Sequential counts as 'other'.\n"
        "empty = nn.Sequential()\n"
        "ec = ex1_count_layer_types(empty)\n"
        "assert ec['other'] == 1 and all(v == 0 for k, v in ec.items() if k != 'other'), f'empty model wrong: {ec}'\n"
        "\n"
        "# Nested Sequential — modules() must recurse.\n"
        "nested = nn.Sequential(\n"
        "    nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8)),\n"
        "    nn.Sequential(nn.ConvTranspose2d(8, 16, 4), nn.BatchNorm2d(16), nn.Linear(16, 4)),\n"
        ")\n"
        "nc = ex1_count_layer_types(nested)\n"
        "assert nc['conv2d'] == 1 and nc['convtranspose2d'] == 1 and nc['batchnorm'] == 2 and nc['linear'] == 1\n"
        "\n"
        "# Mutually-exclusive check: BatchNorm2d must count as 'batchnorm', not 'other'.\n"
        "bn_only = nn.BatchNorm2d(8)\n"
        "bc = ex1_count_layer_types(bn_only)\n"
        "assert bc['batchnorm'] == 1 and bc['other'] == 0, f'BN classified wrong: {bc}'\n"
        "\n"
        "# --- Visualization: bar chart of layer-type counts on a DCGAN-shaped model ---\n"
        "dcgan = nn.Sequential(\n"
        "    nn.ConvTranspose2d(100, 512, 4, stride=1, padding=0, bias=False), nn.BatchNorm2d(512), nn.ReLU(),\n"
        "    nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),\n"
        "    nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),\n"
        "    nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1, bias=False), nn.BatchNorm2d(64),  nn.ReLU(),\n"
        "    nn.ConvTranspose2d(64,  3,   4, stride=2, padding=1, bias=False), nn.Tanh(),\n"
        ")\n"
        "dc = ex1_count_layer_types(dcgan)\n"
        "fig, ax = plt.subplots(figsize=(7, 3))\n"
        "labels = list(dc.keys()); vals = [dc[k] for k in labels]\n"
        "ax.barh(labels, vals, color=['steelblue', 'coral', 'seagreen', 'gold', 'lightgray'])\n"
        "for i, v in enumerate(vals):\n"
        "    ax.text(v + 0.1, i, str(v), va='center')\n"
        "ax.set_xlabel('count')\n"
        "ax.set_title('Layer-type counts in a DCGAN generator')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_count_layer_types(model: nn.Module) -> dict:\n"
        "    counts = {'conv2d': 0, 'convtranspose2d': 0, 'batchnorm': 0, 'linear': 0, 'other': 0}\n"
        "    for m in model.modules():\n"
        "        if isinstance(m, nn.Conv2d):\n"
        "            counts['conv2d'] += 1\n"
        "        elif isinstance(m, nn.ConvTranspose2d):\n"
        "            counts['convtranspose2d'] += 1\n"
        "        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):\n"
        "            counts['batchnorm'] += 1\n"
        "        elif isinstance(m, nn.Linear):\n"
        "            counts['linear'] += 1\n"
        "        else:\n"
        "            counts['other'] += 1\n"
        "    return counts"
    ),
    "solution_notes": (
        "**`model.modules()` is recursive.** It yields the model itself "
        "FIRST, then every descendant — depth-first. For `nn.Sequential(A, "
        "B)` you get `Sequential, A, B`, in that order. That's why the "
        "model + every container shows up under `'other'`.\n\n"
        "**`Conv2d` is NOT a subclass of `ConvTranspose2d`.** They're "
        "siblings under `_ConvNd`. So you can use `if / elif` without "
        "worrying about one matching the other. (If you ever subclass "
        "Conv2d, mind the ordering — check the more specific class first.)\n\n"
        "**Why `elif`, not stacked `if`s.** Mutually exclusive dispatch — "
        "every module gets counted exactly once. Stacked `if`s would "
        "double-count a hypothetical subclass that satisfies two branches."
    ),
    "extra_imports": [
        "import torch.nn as nn",
        "import matplotlib.pyplot as plt",
    ],
})


# -------------------------- ex1 / channel-list-reverse-build
SPECS.append({
    "atom_id": "channel-list-reverse-build",
    "subtopic": "GAN: channel-list reverse build",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_CHANNEL_REVERSE,
    "exercise_index": 1,
    "exercise_title": "build symmetric encoder/decoder channel pairs from a reversed list",
    "slug": "encoder-decoder-channel-list-reverse",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["dcgan", "channels", "symmetry", "list-reverse"],
    "kcs": ["channel-list-reverse-slice", "consecutive-pair-zip"],
    "lo": (
        "Apply slice-reverse `channels[::-1]` and `zip(c[:-1], c[1:])` to "
        "derive symmetric (in_c, out_c) decoder pairs from an encoder "
        "channel list."
    ),
    "prompt_body": (
        "Implement `ex1_encoder_decoder_pairs(encoder_channels)`. The "
        "channel-symmetry pattern at the heart of DCGAN and U-Net "
        "architectures:\n\n"
        "1. `encoder_channels` is a Python list of ints — e.g. `[3, 64, "
        "128, 256, 512]` — read as 'input has 3 channels, after first "
        "block 64, ..., bottleneck 512'.\n"
        "2. Build `encoder_pairs` by walking consecutive elements: zip "
        "`encoder_channels[:-1]` with `encoder_channels[1:]`. Each element "
        "is `(in_c, out_c)`. Return as a list of tuples.\n"
        "3. Build `decoder_channels = encoder_channels[::-1]` (slice "
        "reverse — must be the slice form, not `list(reversed(...))`, so "
        "the structural symmetry shows in the code).\n"
        "4. Build `decoder_pairs` by zipping consecutive elements of "
        "`decoder_channels`.\n"
        "5. Return a dict `{'encoder_pairs': [...], 'decoder_pairs': "
        "[...]}`.\n\n"
        "Input: `encoder_channels` — list of int, length >= 2.\n"
        "Output: dict with two keys, each holding a list of "
        "`(in_c, out_c)` tuples.\n\n"
        "The visualization renders the encoder/decoder channel pyramid as "
        "two stacked bar charts — width-decreasing for the encoder, "
        "width-increasing for the decoder — to show the symmetry."
    ),
    "stub": (
        "def ex1_encoder_decoder_pairs(encoder_channels: list[int]) -> dict:\n"
        '    """Build (in, out) pair lists for encoder and reversed decoder."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Standard DCGAN-ish channel list.\n"
        "enc = [3, 64, 128, 256, 512]\n"
        "out = ex1_encoder_decoder_pairs(enc)\n"
        "assert isinstance(out, dict), f'expected dict, got {type(out).__name__}'\n"
        "assert set(out.keys()) == {'encoder_pairs', 'decoder_pairs'}, f'keys wrong: {set(out.keys())}'\n"
        "\n"
        "expected_enc = [(3, 64), (64, 128), (128, 256), (256, 512)]\n"
        "expected_dec = [(512, 256), (256, 128), (128, 64), (64, 3)]\n"
        "assert out['encoder_pairs'] == expected_enc, f'encoder_pairs wrong: {out[\"encoder_pairs\"]}'\n"
        "assert out['decoder_pairs'] == expected_dec, f'decoder_pairs wrong: {out[\"decoder_pairs\"]}'\n"
        "\n"
        "# Symmetry property — decoder pair i is encoder pair (-1-i) reversed.\n"
        "for i, (in_c, out_c) in enumerate(out['decoder_pairs']):\n"
        "    sym_enc = out['encoder_pairs'][-1 - i]\n"
        "    assert (in_c, out_c) == (sym_enc[1], sym_enc[0]), (\n"
        "        f'symmetry broken at decoder pair {i}: {(in_c, out_c)} vs {sym_enc}'\n"
        "    )\n"
        "\n"
        "# Different list shapes.\n"
        "short = [1, 4]\n"
        "s = ex1_encoder_decoder_pairs(short)\n"
        "assert s['encoder_pairs'] == [(1, 4)] and s['decoder_pairs'] == [(4, 1)]\n"
        "\n"
        "long = [1, 2, 4, 8, 16, 32]\n"
        "l = ex1_encoder_decoder_pairs(long)\n"
        "assert l['encoder_pairs'] == [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32)]\n"
        "assert l['decoder_pairs'] == [(32, 16), (16, 8), (8, 4), (4, 2), (2, 1)]\n"
        "\n"
        "# Input list MUST NOT be mutated.\n"
        "snapshot = enc.copy()\n"
        "ex1_encoder_decoder_pairs(enc)\n"
        "assert enc == snapshot, 'input list must not be mutated'\n"
        "\n"
        "# --- Visualization: encoder pyramid (shrinking spatial) + decoder pyramid (growing spatial) ---\n"
        "viz_enc = [3, 64, 128, 256, 512, 1024]\n"
        "viz_out = ex1_encoder_decoder_pairs(viz_enc)\n"
        "enc_pairs = viz_out['encoder_pairs']; dec_pairs = viz_out['decoder_pairs']\n"
        "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)\n"
        "ax1.bar(range(len(enc_pairs)), [p[1] for p in enc_pairs], color='steelblue', edgecolor='black')\n"
        "for i, (ic, oc) in enumerate(enc_pairs):\n"
        "    ax1.text(i, oc + 20, f'{ic}->{oc}', ha='center', fontsize=9)\n"
        "ax1.set_title('encoder: channel growth')\n"
        "ax1.set_ylabel('out channels')\n"
        "ax2.bar(range(len(dec_pairs)), [p[1] for p in dec_pairs], color='coral', edgecolor='black')\n"
        "for i, (ic, oc) in enumerate(dec_pairs):\n"
        "    ax2.text(i, oc + 20, f'{ic}->{oc}', ha='center', fontsize=9)\n"
        "ax2.set_title('decoder: channel shrink (encoder reversed)')\n"
        "ax2.set_xlabel('block index'); ax2.set_ylabel('out channels')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_encoder_decoder_pairs(encoder_channels: list[int]) -> dict:\n"
        "    encoder_pairs = list(zip(encoder_channels[:-1], encoder_channels[1:]))\n"
        "    decoder_channels = encoder_channels[::-1]\n"
        "    decoder_pairs = list(zip(decoder_channels[:-1], decoder_channels[1:]))\n"
        "    return {'encoder_pairs': encoder_pairs, 'decoder_pairs': decoder_pairs}"
    ),
    "solution_notes": (
        "**`zip(c[:-1], c[1:])` is the consecutive-pair idiom.** It "
        "produces `[(c0, c1), (c1, c2), ..., (c_{n-2}, c_{n-1})]` — the "
        "edges of a path graph over the list. Use this for any "
        "'walk consecutive elements' loop, not just channels.\n\n"
        "**`[::-1]` produces a NEW list.** The slice doesn't mutate the "
        "original, so callers can keep their encoder list intact. (Compare "
        "to `c.reverse()`, which mutates in place — bad idea here.)\n\n"
        "**Why this matters operationally.** In a real network, you'd "
        "iterate `decoder_pairs` to build `nn.Sequential(*[convt_block(ic, "
        "oc) for ic, oc in decoder_pairs])`. The pair list is the contract "
        "between 'I know the channel plan' and 'I build the layers'."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
})


# -------------------------- ex1 / bce-log-loss-real-fake
SPECS.append({
    "atom_id": "bce-log-loss-real-fake",
    "subtopic": "GAN: BCE log loss real/fake",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_BCE_REAL_FAKE,
    "exercise_index": 1,
    "exercise_title": "discriminator BCE loss = BCE(D(real), 1) + BCE(D(fake), 0)",
    "slug": "discriminator-bce-real-plus-fake",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["gan", "bce", "discriminator-loss", "ones-zeros-like"],
    "kcs": ["bce-real-target-1", "bce-fake-target-0-sum"],
    "lo": (
        "Apply `F.binary_cross_entropy` with `t.ones_like` / `t.zeros_like` "
        "targets to compute the discriminator's combined real-plus-fake BCE "
        "loss."
    ),
    "prompt_body": (
        "Implement `ex1_discriminator_loss(d_pred_real, d_pred_fake)`. The "
        "standard DCGAN discriminator loss:\n\n"
        "1. `d_pred_real` are D's probability outputs on REAL images, shape "
        "`(B,)`, values in `[0, 1]` (post-sigmoid).\n"
        "2. `d_pred_fake` are D's probability outputs on FAKE images, shape "
        "`(B,)`, values in `[0, 1]`.\n"
        "3. Build the targets:\n"
        "   - `real_targets = t.ones_like(d_pred_real)` (D wants P=1 on real)\n"
        "   - `fake_targets = t.zeros_like(d_pred_fake)` (D wants P=0 on fake)\n"
        "4. Compute the two BCE terms with `F.binary_cross_entropy(pred, "
        "target)`.\n"
        "5. Return the SUM (not the mean) of the two terms — single scalar.\n\n"
        "Input: `d_pred_real`, `d_pred_fake` — `(B,)` float tensors with "
        "values in `(0, 1)`.\n"
        "Output: scalar tensor.\n\n"
        "The visualization sweeps D's confidence on real images and plots "
        "the loss surface as confidence changes — convex bowl pointing at "
        "perfect classifier."
    ),
    "stub": (
        "def ex1_discriminator_loss(d_pred_real: Tensor, d_pred_fake: Tensor) -> Tensor:\n"
        '    """BCE(D(real), 1) + BCE(D(fake), 0)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn.functional as F\n"
        "import math\n"
        "\n"
        "# Perfect classifier — D=1 on real, D=0 on fake → loss ≈ 0.\n"
        "real = t.full((8,), 0.9999)\n"
        "fake = t.full((8,), 0.0001)\n"
        "loss = ex1_discriminator_loss(real, fake)\n"
        "assert loss.dim() == 0, f'loss must be scalar, got shape {tuple(loss.shape)}'\n"
        "assert loss.item() < 0.001, f'perfect classifier should give loss ~0, got {loss.item():.4f}'\n"
        "\n"
        "# Worst classifier — D=0 on real, D=1 on fake → loss → large.\n"
        "worst_real = t.full((8,), 0.0001)\n"
        "worst_fake = t.full((8,), 0.9999)\n"
        "loss_worst = ex1_discriminator_loss(worst_real, worst_fake)\n"
        "assert loss_worst.item() > 10.0, f'worst classifier should give large loss, got {loss_worst.item():.4f}'\n"
        "\n"
        "# Coin-flip classifier — D=0.5 on both → loss ≈ 2 * log(2) ≈ 1.3863.\n"
        "mid_real = t.full((8,), 0.5)\n"
        "mid_fake = t.full((8,), 0.5)\n"
        "loss_mid = ex1_discriminator_loss(mid_real, mid_fake)\n"
        "expected_mid = 2 * math.log(2)\n"
        "assert abs(loss_mid.item() - expected_mid) < 1e-4, f'coin-flip loss expected {expected_mid:.4f}, got {loss_mid.item():.4f}'\n"
        "\n"
        "# Numerical match against the explicit reference for a random batch.\n"
        "t.manual_seed(0)\n"
        "r = t.rand(16) * 0.8 + 0.1     # in (0.1, 0.9) to avoid log(0)\n"
        "f = t.rand(16) * 0.8 + 0.1\n"
        "got = ex1_discriminator_loss(r, f)\n"
        "expected = F.binary_cross_entropy(r, t.ones_like(r)) + F.binary_cross_entropy(f, t.zeros_like(f))\n"
        "assert t.allclose(got, expected, atol=1e-6), f'numerical mismatch: {got.item()} vs {expected.item()}'\n"
        "\n"
        "# Gradient flows back to D's predictions.\n"
        "r_g = t.full((4,), 0.5, requires_grad=True)\n"
        "f_g = t.full((4,), 0.5, requires_grad=True)\n"
        "ex1_discriminator_loss(r_g, f_g).backward()\n"
        "# d/dr BCE(r, 1) = -1/r so grad w.r.t. r is negative (push r toward 1).\n"
        "assert (r_g.grad < 0).all(), 'gradient on real preds should be negative (push toward 1)'\n"
        "# d/df BCE(f, 0) = 1/(1-f) so grad w.r.t. f is positive (push f toward 0).\n"
        "assert (f_g.grad > 0).all(), 'gradient on fake preds should be positive (push toward 0)'\n"
        "\n"
        "# --- Visualization: loss surface as D's confidence sweeps ---\n"
        "ps = t.linspace(0.01, 0.99, 99)\n"
        "# Hold fake at 0.5; sweep real.\n"
        "losses_real_sweep = [ex1_discriminator_loss(t.full((4,), p.item()), t.full((4,), 0.5)).item() for p in ps]\n"
        "# Hold real at 0.5; sweep fake.\n"
        "losses_fake_sweep = [ex1_discriminator_loss(t.full((4,), 0.5), t.full((4,), p.item())).item() for p in ps]\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "ax.plot(ps.numpy(), losses_real_sweep, label='sweep D(real); D(fake)=0.5', color='steelblue', lw=2)\n"
        "ax.plot(ps.numpy(), losses_fake_sweep, label='sweep D(fake); D(real)=0.5', color='coral', lw=2)\n"
        "ax.axvline(1.0, color='steelblue', ls=':', alpha=0.5)\n"
        "ax.axvline(0.0, color='coral', ls=':', alpha=0.5)\n"
        "ax.set_xlabel('D probability'); ax.set_ylabel('loss')\n"
        "ax.set_title('D loss vs confidence — bowl minima at correct calls')\n"
        "ax.legend(); ax.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_discriminator_loss(d_pred_real: Tensor, d_pred_fake: Tensor) -> Tensor:\n"
        "    import torch.nn.functional as F\n"
        "    loss_real = F.binary_cross_entropy(d_pred_real, t.ones_like(d_pred_real))\n"
        "    loss_fake = F.binary_cross_entropy(d_pred_fake, t.zeros_like(d_pred_fake))\n"
        "    return loss_real + loss_fake"
    ),
    "solution_notes": (
        "**Why sum, not mean across the two terms.** Each BCE is already "
        "MEAN-reduced over the batch (`F.binary_cross_entropy` defaults to "
        "`reduction='mean'`). Summing gives D the full real-loss-plus-fake-"
        "loss signal — the convention dating back to Goodfellow 2014. "
        "Averaging would halve the discriminator's effective learning rate.\n\n"
        "**`ones_like` / `zeros_like` are dtype/device-safe.** They inherit "
        "dtype + device from the prediction tensor — works on GPU, "
        "`bfloat16`, and weird batch shapes without breaking. The naive "
        "`t.ones(d_pred_real.shape[0])` silently breaks on GPU.\n\n"
        "**Numerical caveat.** Bare `F.binary_cross_entropy` can produce "
        "inf when the prediction hits exactly 0 or 1. In production you'd "
        "feed logits to `F.binary_cross_entropy_with_logits` — the fused "
        "version is rock-solid numerically. This recap drills the "
        "probability-input form because that's what flashcards reference."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
})


# -------------------------- ex1 / generator-loss-fool-discriminator
SPECS.append({
    "atom_id": "generator-loss-fool-discriminator",
    "subtopic": "GAN: Generator loss to fool D",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_G_LOSS,
    "exercise_index": 1,
    "exercise_title": "generator BCE loss with target=1 on the fake batch",
    "slug": "generator-bce-fool-discriminator",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["gan", "generator-loss", "non-saturating-bce", "ones-like"],
    "kcs": ["generator-bce-target-1", "non-saturating-formulation"],
    "lo": (
        "Apply `F.binary_cross_entropy(d_pred_fake, t.ones_like(...))` to "
        "compute the non-saturating generator loss — the loss that makes G "
        "push D's verdict toward 1."
    ),
    "prompt_body": (
        "Implement `ex1_generator_loss(d_pred_fake_via_g)`. The non-"
        "saturating generator loss from Goodfellow 2014:\n\n"
        "1. `d_pred_fake_via_g` is D's probability output on the CURRENT "
        "generator's fakes — shape `(B,)`, values in `[0, 1]`.\n"
        "2. Build the targets: `targets = t.ones_like(d_pred_fake_via_g)` "
        "— the generator WANTS D to output 1 on its fakes.\n"
        "3. Return `F.binary_cross_entropy(d_pred_fake_via_g, targets)` "
        "— a scalar.\n\n"
        "That's it — three lines. The subtlety is the asymmetry:\n"
        "- D's loss uses target=0 for fakes ('fake means 0').\n"
        "- G's loss uses target=1 for fakes ('I want D to think fake is "
        "real').\n\n"
        "Same fakes, opposite labels — that's the adversarial signal.\n\n"
        "Input: `d_pred_fake_via_g` — `(B,)` float tensor in `(0, 1)`.\n"
        "Output: scalar tensor.\n\n"
        "The visualization plots G loss as a function of D's confidence on "
        "the fake batch — monotonically decreasing (G is happiest when D "
        "is fooled to confidence 1)."
    ),
    "stub": (
        "def ex1_generator_loss(d_pred_fake_via_g: Tensor) -> Tensor:\n"
        '    """G wants D to predict 1 on fakes → BCE(d_pred, ones)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn.functional as F\n"
        "import math\n"
        "\n"
        "# G fooled D completely — D outputs 1 on fakes → G loss → 0.\n"
        "fooled = t.full((8,), 0.9999)\n"
        "loss_fooled = ex1_generator_loss(fooled)\n"
        "assert loss_fooled.dim() == 0, 'loss must be a scalar'\n"
        "assert loss_fooled.item() < 0.001, f'fully-fooled D should give G loss ~0, got {loss_fooled.item():.5f}'\n"
        "\n"
        "# G failed completely — D outputs 0 on fakes → G loss → large.\n"
        "failed = t.full((8,), 0.0001)\n"
        "loss_failed = ex1_generator_loss(failed)\n"
        "assert loss_failed.item() > 5.0, f'fully-detected fakes should give large G loss, got {loss_failed.item():.4f}'\n"
        "\n"
        "# Coin-flip D — G loss = log(2) ≈ 0.6931.\n"
        "coin = t.full((8,), 0.5)\n"
        "loss_coin = ex1_generator_loss(coin)\n"
        "expected_coin = math.log(2)\n"
        "assert abs(loss_coin.item() - expected_coin) < 1e-4, f'coin-flip G loss expected {expected_coin:.4f}, got {loss_coin.item():.4f}'\n"
        "\n"
        "# Numerical match against reference.\n"
        "t.manual_seed(0)\n"
        "preds = t.rand(16) * 0.8 + 0.1\n"
        "got = ex1_generator_loss(preds)\n"
        "expected = F.binary_cross_entropy(preds, t.ones_like(preds))\n"
        "assert t.allclose(got, expected, atol=1e-6), f'numerical mismatch: {got.item()} vs {expected.item()}'\n"
        "\n"
        "# Gradient pushes D-prob UP (i.e. grad on d_pred is negative).\n"
        "p = t.full((4,), 0.5, requires_grad=True)\n"
        "ex1_generator_loss(p).backward()\n"
        "# d/dp BCE(p, 1) = -1/p < 0\n"
        "assert (p.grad < 0).all(), 'gradient should be negative (push d_pred toward 1)'\n"
        "\n"
        "# --- Visualization: G loss vs D's confidence on fakes ---\n"
        "ps = t.linspace(0.01, 0.99, 99)\n"
        "losses = [ex1_generator_loss(t.full((4,), p.item())).item() for p in ps]\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "ax.plot(ps.numpy(), losses, color='seagreen', lw=2)\n"
        "ax.axvline(1.0, color='red', ls=':', label='G goal: D(fake)=1')\n"
        "ax.set_xlabel('D probability on fakes'); ax.set_ylabel('generator loss')\n"
        "ax.set_title('non-saturating G loss — monotonically decreasing in D(fake)')\n"
        "ax.legend(); ax.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_generator_loss(d_pred_fake_via_g: Tensor) -> Tensor:\n"
        "    import torch.nn.functional as F\n"
        "    targets = t.ones_like(d_pred_fake_via_g)\n"
        "    return F.binary_cross_entropy(d_pred_fake_via_g, targets)"
    ),
    "solution_notes": (
        "**Non-saturating vs saturating.** The paper-original `min_G "
        "log(1 - D(G(z)))` saturates — when D is confident the fake is "
        "fake (D(fake) ≈ 0), `log(1 - D(fake)) ≈ log(1) ≈ 0` and the "
        "gradient w.r.t. G vanishes. The flipped target `BCE(D(fake), 1) "
        "= -log(D(fake))` instead goes to infinity in the same regime — "
        "strong gradient when G needs it most. Goodfellow's own footnote.\n\n"
        "**G only sees D's verdict, not D's params.** The loss depends "
        "on `d_pred_fake_via_g` (a tensor produced by D), so autograd "
        "would normally try to flow gradient through D too. The training "
        "loop handles that by stepping ONLY `optim_G` after computing "
        "G's loss — `optim_G.param_groups` contain only G's parameters.\n\n"
        "**One-liner is correct.** No need for ones-tensor construction "
        "logic — `ones_like` gives you the right shape/dtype/device in a "
        "single token. The whole loss really is three meaningful lines."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
})


# -------------------------- ex1 / noise-batch-from-latent
SPECS.append({
    "atom_id": "noise-batch-from-latent",
    "subtopic": "GAN: Noise batch from latent_dim",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_NOISE_BATCH,
    "exercise_index": 1,
    "exercise_title": "build (B, latent_dim, 1, 1) spatial-prefix noise for DCGAN G",
    "slug": "noise-batch-spatial-prefix",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["dcgan", "noise", "randn", "spatial-prefix"],
    "kcs": ["randn-4d-shape", "spatial-1x1-prefix"],
    "lo": (
        "Apply `t.randn(B, latent_dim, 1, 1)` to construct the 4-D Gaussian "
        "noise batch that DCGAN generators accept as input."
    ),
    "prompt_body": (
        "Implement `ex1_dcgan_noise_batch(batch_size, latent_dim, "
        "generator)`. The DCGAN generator input — 4-D noise tensor with "
        "trailing 1×1 spatial dims:\n\n"
        "1. Sample standard-normal noise of shape `(batch_size, latent_dim, "
        "1, 1)`.\n"
        "2. Use `t.randn(batch_size, latent_dim, 1, 1, generator=generator)` "
        "— pass the generator through so the call is reproducible.\n"
        "3. Return the resulting tensor (dtype `float32`).\n\n"
        "Important constraints:\n"
        "- The output must be 4-D (`x.ndim == 4`), not 2-D. A 2-D flat "
        "vector won't pass through a `ConvTranspose2d(latent_dim, ..., "
        "kernel_size=4)`.\n"
        "- The trailing two dims must be EXACTLY 1 — the first transposed "
        "conv expects a 1×1 seed.\n"
        "- Use `t.randn` (not `t.rand`); the DCGAN prior is `N(0, 1)`, "
        "not uniform.\n\n"
        "Input: `batch_size` int, `latent_dim` int, `generator` "
        "`torch.Generator` for seed reproducibility.\n"
        "Output: `(batch_size, latent_dim, 1, 1)` float32 tensor.\n\n"
        "The visualization renders the noise as a `(B × latent_dim)` "
        "heatmap (collapsing the 1×1 spatial axes) so you can verify the "
        "shape contract."
    ),
    "stub": (
        "def ex1_dcgan_noise_batch(batch_size: int, latent_dim: int, generator: 'torch.Generator') -> Tensor:\n"
        '    """Standard-normal noise of shape (B, latent_dim, 1, 1)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "\n"
        "# Shape contract — must be (B, latent_dim, 1, 1).\n"
        "rng = t.Generator().manual_seed(0)\n"
        "noise = ex1_dcgan_noise_batch(8, 100, rng)\n"
        "assert noise.shape == (8, 100, 1, 1), f'expected (8, 100, 1, 1), got {tuple(noise.shape)}'\n"
        "assert noise.dtype == t.float32\n"
        "assert noise.ndim == 4, f'noise must be 4-D, got ndim={noise.ndim}'\n"
        "\n"
        "# Distribution sanity — sample a large batch and check ~N(0, 1).\n"
        "rng2 = t.Generator().manual_seed(0)\n"
        "big = ex1_dcgan_noise_batch(2000, 100, rng2)\n"
        "assert abs(big.mean().item()) < 0.05, f'noise mean should be ~0, got {big.mean().item():.4f}'\n"
        "assert abs(big.std().item() - 1.0) < 0.05, f'noise std should be ~1, got {big.std().item():.4f}'\n"
        "\n"
        "# Reproducibility — same seed → same noise.\n"
        "rng_a = t.Generator().manual_seed(42)\n"
        "rng_b = t.Generator().manual_seed(42)\n"
        "n_a = ex1_dcgan_noise_batch(4, 8, rng_a)\n"
        "n_b = ex1_dcgan_noise_batch(4, 8, rng_b)\n"
        "assert t.equal(n_a, n_b), 'same seed must produce same noise'\n"
        "\n"
        "# Shape must be valid input to a real DCGAN first layer.\n"
        "first_layer = nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, padding=0, bias=False)\n"
        "rng3 = t.Generator().manual_seed(0)\n"
        "noise_for_layer = ex1_dcgan_noise_batch(2, 100, rng3)\n"
        "out = first_layer(noise_for_layer)\n"
        "assert out.shape == (2, 512, 4, 4), f'first layer output wrong: {tuple(out.shape)}'\n"
        "\n"
        "# Different latent_dim works.\n"
        "rng4 = t.Generator().manual_seed(0)\n"
        "small_lat = ex1_dcgan_noise_batch(4, 16, rng4)\n"
        "assert small_lat.shape == (4, 16, 1, 1)\n"
        "\n"
        "# --- Visualization: noise heatmap (B × latent_dim, collapsing 1×1) ---\n"
        "rng_v = t.Generator().manual_seed(7)\n"
        "viz_noise = ex1_dcgan_noise_batch(16, 64, rng_v)\n"
        "viz_flat = viz_noise.squeeze(-1).squeeze(-1)   # (16, 64)\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "im = ax.imshow(viz_flat.numpy(), aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)\n"
        "ax.set_xlabel('latent dim'); ax.set_ylabel('batch idx')\n"
        "ax.set_title(f'noise batch (B=16, latent=64) — shape after squeeze: {tuple(viz_flat.shape)}')\n"
        "plt.colorbar(im, ax=ax, fraction=0.046, label='noise value')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_dcgan_noise_batch(batch_size: int, latent_dim: int, generator: 'torch.Generator') -> Tensor:\n"
        "    return t.randn(batch_size, latent_dim, 1, 1, generator=generator)"
    ),
    "solution_notes": (
        "**Why the trailing 1×1.** The DCGAN generator is fully "
        "convolutional. Its first `ConvTranspose2d(latent_dim, ..., "
        "kernel_size=4, stride=1, padding=0)` blows a `1×1` seed up to "
        "`4×4` (output shape = `(1-1)*1 - 2*0 + 4 = 4`). Without the "
        "spatial prefix the conv has no spatial axis to grow from.\n\n"
        "**Why `t.randn`, not `t.rand`.** Standard normal vs uniform — "
        "different priors give different sample distributions. DCGAN's "
        "prior is `N(0, I)`; flipping to `Uniform(0, 1)` will train but "
        "produces visibly worse samples and mismatched latent-space "
        "interpolation behavior.\n\n"
        "**`generator=` for reproducibility.** Passing an explicit "
        "`torch.Generator` lets the caller control the RNG without "
        "touching the global state. Critical for unit tests, training "
        "reproducibility, and seeding multi-GPU jobs."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
})


# -------------------------- ex1 / model-train-eval-toggle-around-sample
SPECS.append({
    "atom_id": "model-train-eval-toggle-around-sample",
    "subtopic": "GAN: model.train/eval toggle around sample",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_TRAIN_EVAL_TOGGLE,
    "exercise_index": 1,
    "exercise_title": "eval→no_grad→sample→train: clean sampling inside a training loop",
    "slug": "train-eval-toggle-around-sample",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["gan", "eval-mode", "no_grad", "batchnorm-running-stats"],
    "kcs": ["model-eval-no-grad-sample", "restore-train-mode"],
    "lo": (
        "Apply the `eval() → no_grad() → forward → train()` toggle pattern "
        "to sample from a generator inside a training loop without "
        "corrupting BatchNorm running statistics."
    ),
    "prompt_body": (
        "Implement `ex1_sample_clean(model, noise)`. The standard "
        "BN-safe sampling block:\n\n"
        "1. Switch the model to evaluation mode: `model.eval()`.\n"
        "2. Enter a `torch.no_grad()` context. INSIDE the context, run "
        "the forward pass: `samples = model(noise)`.\n"
        "3. Switch the model BACK to training mode: `model.train()`.\n"
        "4. Return `samples`.\n\n"
        "Critical invariants the test checks:\n"
        "- `samples.requires_grad` must be `False` (because of `no_grad`).\n"
        "- `model.training` must be `True` AFTER the call (we restored it).\n"
        "- The model's `BatchNorm.running_mean` and `running_var` must be "
        "UNCHANGED by the sampling call (because `eval()` makes BN use "
        "running stats and skip the update).\n\n"
        "Input: `model` — `nn.Module` with BatchNorm layers somewhere; "
        "`noise` — input tensor of the right shape.\n"
        "Output: model output tensor, with `requires_grad=False`, the "
        "model left in `train()` mode, BN running stats untouched.\n\n"
        "The visualization plots BatchNorm's running_mean across the layer "
        "BEFORE and AFTER a sample call to confirm it didn't drift."
    ),
    "stub": (
        "def ex1_sample_clean(model: nn.Module, noise: Tensor) -> Tensor:\n"
        '    """Sample without corrupting BN running stats: eval → no_grad → forward → train."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "\n"
        "# Build a small DCGAN-ish generator with BN.\n"
        "model = nn.Sequential(\n"
        "    nn.ConvTranspose2d(10, 32, 4, stride=1, padding=0, bias=False),\n"
        "    nn.BatchNorm2d(32),\n"
        "    nn.ReLU(),\n"
        "    nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=False),\n"
        "    nn.Tanh(),\n"
        ")\n"
        "# Prime BN running stats with a few real-data forward passes in train mode.\n"
        "model.train()\n"
        "for _ in range(3):\n"
        "    _ = model(t.randn(8, 10, 1, 1))\n"
        "bn_layer = model[1]\n"
        "rmean_before = bn_layer.running_mean.detach().clone()\n"
        "rvar_before = bn_layer.running_var.detach().clone()\n"
        "\n"
        "# Now sample with a wildly different distribution → would shift BN stats if eval() were missing.\n"
        "weird_noise = 10.0 * t.randn(8, 10, 1, 1)\n"
        "out = ex1_sample_clean(model, weird_noise)\n"
        "\n"
        "# Shape sanity — output should be image-shaped.\n"
        "assert out.shape == (8, 3, 8, 8), f'unexpected output shape {tuple(out.shape)}'\n"
        "\n"
        "# requires_grad must be False — no_grad context.\n"
        "assert not out.requires_grad, 'samples should NOT require grad (no_grad context)'\n"
        "\n"
        "# Model must be left in training mode.\n"
        "assert model.training, 'model must be restored to training mode'\n"
        "for m in model.modules():\n"
        "    if isinstance(m, nn.BatchNorm2d):\n"
        "        assert m.training, f'BatchNorm submodule must also be back in training mode'\n"
        "\n"
        "# BN running stats must NOT have changed during sampling.\n"
        "assert t.allclose(bn_layer.running_mean, rmean_before), 'BN running_mean must NOT have changed during sample'\n"
        "assert t.allclose(bn_layer.running_var, rvar_before), 'BN running_var must NOT have changed during sample'\n"
        "\n"
        "# Sanity — a normal forward pass (in train mode, no toggle) DOES update running stats.\n"
        "rmean_pre_train = bn_layer.running_mean.detach().clone()\n"
        "_ = model(t.randn(8, 10, 1, 1))\n"
        "assert not t.allclose(bn_layer.running_mean, rmean_pre_train), (\n"
        "    'Sanity broke: plain train-mode forward should update BN running_mean. '\n"
        "    'If this fires, the model has no live BN layers, not your bug.'\n"
        ")\n"
        "\n"
        "# Output value match — eval-mode forward + same noise gives reproducible result.\n"
        "model.train()\n"
        "rng = t.Generator().manual_seed(0)\n"
        "fixed_noise = t.randn(2, 10, 1, 1, generator=rng)\n"
        "s1 = ex1_sample_clean(model, fixed_noise)\n"
        "s2 = ex1_sample_clean(model, fixed_noise)\n"
        "assert t.allclose(s1, s2), 'two eval-mode samples on same noise must match (deterministic)'\n"
        "\n"
        "# --- Visualization: BN running_mean before vs after a sample call ---\n"
        "fig, ax = plt.subplots(figsize=(8, 3))\n"
        "ax.plot(rmean_before.numpy(), 'o-', label='before sample', color='steelblue', alpha=0.8)\n"
        "ax.plot(bn_layer.running_mean.numpy(), 'x--', label='after sample (should match)', color='coral', alpha=0.8)\n"
        "ax.set_xlabel('BN channel idx'); ax.set_ylabel('running_mean')\n"
        "ax.set_title('BN running_mean — unchanged after eval-mode sample')\n"
        "ax.legend(); ax.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_sample_clean(model: nn.Module, noise: Tensor) -> Tensor:\n"
        "    model.eval()\n"
        "    with t.no_grad():\n"
        "        samples = model(noise)\n"
        "    model.train()\n"
        "    return samples"
    ),
    "solution_notes": (
        "**`eval()` propagates to submodules.** Calling `model.eval()` "
        "recursively switches every BatchNorm and Dropout submodule. Same "
        "for `train()`. You don't have to walk the children yourself.\n\n"
        "**Why two mechanisms (mode + no_grad).** They're orthogonal. "
        "`model.eval()` is about RUNTIME BEHAVIOR — BN uses running stats, "
        "Dropout is a no-op. `torch.no_grad()` is about AUTOGRAD — no "
        "graph is built, `requires_grad` propagation is suppressed. You "
        "need eval() for correctness; you need no_grad() for memory + "
        "speed.\n\n"
        "**Forgetting `model.train()` is the canonical bug.** The next "
        "real training step happens with BN frozen on running stats, and "
        "with running stats that aren't being updated. The training loss "
        "subtly diverges. Always restore — even better, use a `try/"
        "finally` if you've got side effects between."
    ),
    "extra_imports": [
        "import torch.nn as nn",
        "import matplotlib.pyplot as plt",
    ],
})


# ---------------------------------------------------------------- emit + verify

for spec in SPECS:
    path = emit_standalone(spec)
    rel = path.relative_to(path.parents[3])
    print(f"wrote {rel}")
