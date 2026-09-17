#!/usr/bin/env python3
"""Author Colab-native standalones for ARENA chapter-0 part-5 (VAE / GAN) atoms.

Batch 5: eight single-exercise drills, brand-new folder `prereqs_generative/`.
Each atom gets ONE ex (ex1). Each exercise hits ONE Bloom level + at most 2 KCs.

Atom roster (8):
    - bottleneck-latent-projection      (Linear (B,784)->(B,latent) projection)
    - mse-reconstruction-loss           (F.mse_loss vs per-pixel / per-sample mean)
    - holdout-data-one-per-class        (one-sample-per-class gallery selection)
    - randn-like-noise-source           (randn_like preserves device + dtype)
    - dcgan-wrapper-netG-netD           (single Module holds netG + netD as submodules)
    - broadcast-source-fanout           (class embedding fan-out via labels indexing)
    - t-stack-trajectory                (stack list of (B,D) -> (B,T,D) trajectory)
    - requires-grad-leaf-assert         (assert p.is_leaf at optimizer init)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_generative"


# ---------------------------------------------------------------- recaps

RECAP_BOTTLENECK = (
    "## Bottleneck latent projection — quick refresher\n"
    "\n"
    "The bottleneck is the SINGLE narrowest layer in an autoencoder. It is "
    "almost always a `nn.Linear(flattened_dim, latent_dim)` where `latent_dim "
    "<< flattened_dim`. Two non-obvious things:\n"
    "\n"
    "1. **Dim reduction is the entire job.** No activation, no normalization "
    "lives at the bottleneck itself — just `Linear`. Any non-linearity goes "
    "BEFORE the projection (in the encoder body) or AFTER (in the decoder "
    "body), never on the bottleneck output.\n"
    "2. **Input is already flat.** ARENA flattens with "
    "`Rearrange('b c h w -> b (c h w)')` before the bottleneck Linear. The "
    "bottleneck does not know about spatial structure — it sees a 1-D feature "
    "vector per batch element.\n"
    "\n"
    "**Why it matters.** Visualizing the latent space (2-D scatter colored by "
    "class label) tells you whether the bottleneck has learned a useful "
    "compression. A well-trained autoencoder produces class-clustered "
    "latents; a random-weights one produces a featureless blob."
)

RECAP_MSE = (
    "## MSE reconstruction loss — quick refresher\n"
    "\n"
    "Autoencoders train against `loss = F.mse_loss(decoded, original)`. The "
    "default `reduction='mean'` averages over EVERY element — batch, channel, "
    "height, width — giving a single scalar.\n"
    "\n"
    "**Three reductions to know.**\n"
    "- `reduction='mean'` (default) — scalar; mean over `B*C*H*W` elements.\n"
    "- `reduction='sum'` — scalar; sum over all elements (B times bigger than "
    "mean for fixed batch).\n"
    "- `reduction='none'` — per-element `(B,C,H,W)` tensor; you decide how "
    "to reduce.\n"
    "\n"
    "**Per-sample loss.** When you want one loss number per batch element "
    "(e.g. to weight some images more heavily): use `reduction='none'`, then "
    "`.mean(dim=[1,2,3])` to average within each sample. This gives you a "
    "`(B,)` vector.\n"
    "\n"
    "**Argument order.** `F.mse_loss(input, target)` — both arguments are "
    "symmetric (MSE is symmetric in its inputs), so the order doesn't change "
    "the value. But by convention `input` is the model output, `target` is "
    "the ground truth."
)

RECAP_HOLDOUT = (
    "## Hold-out one-per-class data — quick refresher\n"
    "\n"
    "When logging autoencoder / VAE reconstructions during training you want "
    "a FIXED, REPRESENTATIVE batch — usually one image per class — so the "
    "visualization tells the same story across epochs.\n"
    "\n"
    "The pattern:\n"
    "```python\n"
    "holdout = []\n"
    "for c in range(num_classes):\n"
    "    mask = labels == c\n"
    "    holdout.append(data[mask][0])    # FIRST sample of class c\n"
    "holdout = t.stack(holdout, dim=0)    # (num_classes, *sample_shape)\n"
    "```\n"
    "\n"
    "**Why ONE per class.** Cheaper to log, easier to read on a 1×K grid, "
    "and guarantees every class is represented even if the test set is "
    "imbalanced. The first hit of each class is fine — you don't need a "
    "random pick because the dataset shuffle is upstream.\n"
    "\n"
    "**Mask indexing returns a copy.** `data[mask]` always allocates fresh "
    "storage. Take `[0]` rather than `[:1]` if you want a non-batch shape; "
    "the final `t.stack` re-adds the leading axis."
)

RECAP_RANDN_LIKE = (
    "## `randn_like` noise source — quick refresher\n"
    "\n"
    "`t.randn_like(x)` returns a standard-normal tensor with the SAME shape, "
    "dtype, and device as `x`. It is the canonical noise source for "
    "VAE reparameterization (`z = mu + sigma * t.randn_like(sigma)`).\n"
    "\n"
    "**Compared to `t.randn(*x.shape)`.** That call also matches shape, but "
    "produces a `float32` tensor on the CPU — silently breaks when `x` is on "
    "GPU or half-precision. Symptoms: `RuntimeError: expected all tensors to "
    "be on the same device` or huge numerical drift from a float32-on-fp16 "
    "graph.\n"
    "\n"
    "**Rule of thumb.** Any time the noise needs to be added to / multiplied "
    "by an existing tensor `x`, use `randn_like(x)` — it inherits all three "
    "of (shape, dtype, device) so the downstream op is always well-typed."
)

RECAP_DCGAN_WRAPPER = (
    "## DCGAN wrapper module — quick refresher\n"
    "\n"
    "ARENA's `DCGAN` class is a *wrapper* — a single `nn.Module` that holds "
    "BOTH the generator and the discriminator as submodules:\n"
    "\n"
    "```python\n"
    "class DCGAN(nn.Module):\n"
    "    def __init__(self, ...):\n"
    "        super().__init__()\n"
    "        self.netG = Generator(...)\n"
    "        self.netD = Discriminator(...)\n"
    "```\n"
    "\n"
    "**No `forward` method.** Callers invoke `model.netG(noise)` and "
    "`model.netD(img)` directly — there isn't a single sensible signature "
    "for the joint forward (one takes noise, the other takes images), so we "
    "skip it.\n"
    "\n"
    "**Why bother with a wrapper at all?** Two reasons:\n"
    "1. `.to(device)` on the wrapper moves both networks in one call.\n"
    "2. `state_dict()` snapshots BOTH at once for clean checkpointing.\n"
    "\n"
    "**`parameters()` gotcha.** `dcgan.parameters()` returns ALL params from "
    "BOTH subnets — never pass that straight to an optimizer in a GAN. You "
    "need TWO optimizers (`Adam(dcgan.netG.parameters(), ...)` and "
    "`Adam(dcgan.netD.parameters(), ...)`) so the gradient steps stay "
    "adversarial."
)

RECAP_BROADCAST_FANOUT = (
    "## Broadcast source fan-out — quick refresher\n"
    "\n"
    "Class-conditional models hold ONE embedding per class — a `(num_classes, "
    "D)` parameter matrix. To turn that into a per-batch embedding, you "
    "index by labels:\n"
    "\n"
    "```python\n"
    "embed_table = nn.Parameter(t.randn(num_classes, D))\n"
    "per_sample = embed_table[labels]   # labels: (B,) → per_sample: (B, D)\n"
    "```\n"
    "\n"
    "**This IS broadcasting.** A `(num_classes, D)` source tensor gets fanned "
    "out to a `(B, D)` tensor with possibly-repeated rows. Two labels that "
    "are equal get the same embedding row — that's what makes it "
    "*class-conditional*.\n"
    "\n"
    "**Equivalent forms.** `embed_table[labels]` is identical to "
    "`F.embedding(labels, embed_table)` is identical to "
    "`nn.Embedding(num_classes, D)(labels)`. The Module form gets you "
    "`.weight` as a `nn.Parameter` automatically; the raw-index form is "
    "fine when you already have the table as a Parameter elsewhere.\n"
    "\n"
    "**Gradient flow.** Gradients flow ONLY to the rows that were indexed — "
    "embedding lookup is differentiable in a sparse way. Classes absent from "
    "the batch receive zero gradient that step."
)

RECAP_T_STACK_TRAJECTORY = (
    "## `torch.stack` trajectory — quick refresher\n"
    "\n"
    "When you want to keep a TIME SERIES of intermediate tensors (per-step "
    "latents, per-epoch losses, per-iteration samples), the idiom is:\n"
    "\n"
    "```python\n"
    "history = []                          # python list\n"
    "for step in range(T):\n"
    "    z = compute_latent_at_step(step)  # (B, D)\n"
    "    history.append(z)\n"
    "trajectory = t.stack(history, dim=1)  # (B, T, D)\n"
    "```\n"
    "\n"
    "**Why `dim=1`, not `dim=0`.** `dim=0` produces `(T, B, D)` — time is "
    "the outermost axis. `dim=1` produces `(B, T, D)` — preserves batch as "
    "the outermost, time becomes the second axis. The latter is what most "
    "downstream tools expect (it matches "
    "`(batch, sequence, features)` from RNN/transformer conventions).\n"
    "\n"
    "**Stack vs cat.** `t.stack` introduces a NEW axis at `dim`. `t.cat` "
    "concatenates along an EXISTING axis. If each `z` is `(B, D)` and you "
    "want `(B, T, D)`, you must `stack` — `cat(history, dim=1)` would give "
    "`(B, T*D)` instead.\n"
    "\n"
    "**All tensors must match shape and dtype.** `t.stack` raises if they "
    "don't. Easy bug: appending a `(B, D)` and a `(D,)` from the same loop "
    "by accident."
)

RECAP_REQ_GRAD_LEAF = (
    "## `requires_grad` + leaf assertion — quick refresher\n"
    "\n"
    "Optimizers train *leaf parameters* — tensors created by `nn.Parameter(...)` "
    "or `t.tensor(..., requires_grad=True)`. A *non-leaf* tensor is one produced "
    "by an op (e.g. `param * 2`) — it carries gradients during backward but "
    "`opt.step()` won't update it (it has no `.data` storage of its own).\n"
    "\n"
    "**The defensive pattern.** Before passing a list of params to "
    "`Adam([...], lr=...)`, assert:\n"
    "```python\n"
    "for p in params:\n"
    "    assert p.is_leaf, f'{p.shape} is non-leaf — optimizer will silently skip it'\n"
    "    assert p.requires_grad, f'{p.shape} has requires_grad=False — will not update'\n"
    "```\n"
    "\n"
    "**Why this saves hours.** A common bug: you `.to(device)` a single "
    "Parameter (instead of the whole Module). The result is a NEW non-leaf "
    "tensor that the optimizer accepts, runs `.step()` on without error, "
    "and silently does nothing. Training loss plateaus, you spend a day "
    "looking at the model — the bug is in the optimizer init.\n"
    "\n"
    "**Module-level fix.** Build your `nn.Module`, move the WHOLE module "
    "with `model.to(device)`, then pass `model.parameters()` to the "
    "optimizer. Every yielded tensor is a leaf by construction."
)


# ---------------------------------------------------------------- specs

SPECS = []

# -------------------------- ex1 / bottleneck-latent-projection
SPECS.append({
    "atom_id": "bottleneck-latent-projection",
    "subtopic": "Generative: Bottleneck latent projection",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_BOTTLENECK,
    "exercise_index": 1,
    "exercise_title": "encode flat batch into latent + 2-D scatter",
    "slug": "encode-flat-batch-into-latent",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["bottleneck", "Linear", "latent-projection", "visualization"],
    "kcs": ["bottleneck-linear-projection", "bottleneck-dim-reduction"],
    "lo": (
        "Apply a single `nn.Linear(input_dim, latent_dim)` to project a flat "
        "`(B, 784)` batch into a `(B, latent_dim)` latent code, with no "
        "non-linearity on the bottleneck output."
    ),
    "prompt_body": (
        "Implement `ex1_bottleneck_project(flat_batch, weight, bias)`. The "
        "ATOMIC bottleneck operation that every autoencoder hides inside "
        "its encoder:\n\n"
        "1. `flat_batch` has shape `(B, 784)` — already flattened MNIST.\n"
        "2. `weight` has shape `(latent_dim, 784)` and `bias` has shape "
        "`(latent_dim,)` — together they parameterize an `nn.Linear(784, "
        "latent_dim)` (PyTorch stores `weight` as `(out_features, in_features)`).\n"
        "3. Compute `latent = flat_batch @ weight.T + bias`.\n"
        "4. **DO NOT** apply ReLU, sigmoid, layer-norm, or anything else — "
        "the bottleneck output is the bare Linear projection.\n\n"
        "Input: `(B, 784)` float tensor.\n"
        "Output: `(B, latent_dim)` float tensor.\n\n"
        "The visualization runs your function on a synthetic 3-class dataset "
        "with `latent_dim=2` and scatters the latent codes colored by class — "
        "well-chosen weights produce three visible clusters."
    ),
    "stub": (
        "def ex1_bottleneck_project(flat_batch: Tensor, weight: Tensor, bias: Tensor) -> Tensor:\n"
        '    """Project (B, 784) -> (B, latent_dim) via a single Linear."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Smoke test — known shapes.\n"
        "B, in_dim, latent_dim = 4, 784, 5\n"
        "flat = t.randn(B, in_dim, generator=t.Generator().manual_seed(0))\n"
        "W = t.randn(latent_dim, in_dim, generator=t.Generator().manual_seed(1))\n"
        "b = t.randn(latent_dim, generator=t.Generator().manual_seed(2))\n"
        "out = ex1_bottleneck_project(flat, W, b)\n"
        "assert out.shape == (B, latent_dim), f'expected (B,latent_dim)=({B},{latent_dim}), got {tuple(out.shape)}'\n"
        "assert out.dtype == t.float32, f'expected float32, got {out.dtype}'\n"
        "\n"
        "# Numerical correctness — must equal the bare affine.\n"
        "expected = flat @ W.T + b\n"
        "assert t.allclose(out, expected, atol=1e-5), 'output must equal flat @ W.T + b'\n"
        "\n"
        "# Identity case — W=I (square), b=0 → should reproduce the input.\n"
        "id_W = t.eye(in_dim)\n"
        "id_b = t.zeros(in_dim)\n"
        "out_id = ex1_bottleneck_project(flat, id_W, id_b)\n"
        "assert t.allclose(out_id, flat, atol=1e-5), 'identity projection must be a no-op'\n"
        "\n"
        "# Catch: a ReLU on the output would clip negatives — guard against it.\n"
        "neg_flat = -t.ones(2, in_dim)\n"
        "neg_W = t.eye(in_dim)\n"
        "neg_b = t.zeros(in_dim)\n"
        "out_neg = ex1_bottleneck_project(neg_flat, neg_W, neg_b)\n"
        "assert (out_neg < 0).all(), 'bottleneck has NO ReLU — negatives must pass through'\n"
        "\n"
        "# --- Visualization: 2-D latents colored by class ---\n"
        "rng = t.Generator().manual_seed(7)\n"
        "K = 3                  # classes\n"
        "Nc = 60                # samples per class\n"
        "in_dim_viz = 784\n"
        "latent_dim_viz = 2\n"
        "# Each class has a different mean in input space.\n"
        "centers = t.randn(K, in_dim_viz, generator=rng)\n"
        "labels = t.cat([t.full((Nc,), c, dtype=t.long) for c in range(K)])\n"
        "data = t.cat([centers[c].unsqueeze(0) + 0.3 * t.randn(Nc, in_dim_viz, generator=rng) for c in range(K)], dim=0)\n"
        "# Weights chosen to separate centers in 2-D.\n"
        "W_viz = t.stack([centers[0] - centers[1], centers[0] - centers[2]], dim=0)  # (2, 784)\n"
        "b_viz = t.zeros(latent_dim_viz)\n"
        "latents = ex1_bottleneck_project(data, W_viz, b_viz)\n"
        "fig, ax = plt.subplots(figsize=(5, 5))\n"
        "for c in range(K):\n"
        "    mask = labels == c\n"
        "    ax.scatter(latents[mask, 0].numpy(), latents[mask, 1].numpy(),\n"
        "               label=f'class {c}', alpha=0.7, s=25)\n"
        "ax.set_xlabel('latent dim 0')\n"
        "ax.set_ylabel('latent dim 1')\n"
        "ax.set_title('ex1 bottleneck latents (2-D scatter, colored by class)')\n"
        "ax.legend()\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_bottleneck_project(flat_batch: Tensor, weight: Tensor, bias: Tensor) -> Tensor:\n"
        "    return flat_batch @ weight.T + bias"
    ),
    "solution_notes": (
        "**Why `weight.T`.** PyTorch stores `nn.Linear` weights as "
        "`(out_features, in_features)` so that `Linear(in, out).weight` has "
        "shape `(out, in)` — that's the row-per-output convention used in "
        "papers. To apply it to `(B, in)` input you need the transpose so "
        "the contraction is `(B, in) @ (in, out) → (B, out)`.\n\n"
        "**The 2-D scatter is the diagnostic.** A bottleneck whose latents "
        "blob into one cluster has learned nothing. A bottleneck whose "
        "latents form K distinct clusters has implicitly learned a class-"
        "separating compression — without ever being trained on labels. "
        "That's the magic of autoencoders.\n\n"
        "**No activation on the bottleneck.** Adding ReLU here would clip "
        "negative latents — half the representation space — for no benefit. "
        "Adding BatchNorm or LayerNorm would collapse the very dim "
        "reduction you're trying to measure."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
})


# -------------------------- ex1 / mse-reconstruction-loss
SPECS.append({
    "atom_id": "mse-reconstruction-loss",
    "subtopic": "Generative: MSE reconstruction loss",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_MSE,
    "exercise_index": 1,
    "exercise_title": "scalar mse loss + side-by-side reconstructions",
    "slug": "scalar-mse-loss",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["mse", "reconstruction", "F.mse_loss", "visualization"],
    "kcs": ["mse-default-reduction-mean", "mse-per-sample-via-none"],
    "lo": (
        "Apply `F.mse_loss` with the correct `reduction` arg to compute (a) "
        "the default scalar mean over all elements and (b) a `(B,)` "
        "per-sample loss vector."
    ),
    "prompt_body": (
        "Implement `ex1_mse_losses(original, decoded)`. Two reductions, one "
        "function call each:\n\n"
        "1. `original` and `decoded` both have shape `(B, 1, 28, 28)` — a "
        "minibatch of MNIST images.\n"
        "2. Compute `scalar_loss = F.mse_loss(decoded, original)` — uses the "
        "default `reduction='mean'`, returns a 0-D scalar.\n"
        "3. Compute `per_sample_loss` by calling `F.mse_loss` with "
        "`reduction='none'` to get a `(B, 1, 28, 28)` per-element tensor, "
        "then averaging over the last three axes to collapse it to `(B,)`.\n"
        "4. Return a tuple `(scalar_loss, per_sample_loss)`.\n\n"
        "Input: `original`, `decoded` — `(B, 1, 28, 28)` float tensors.\n"
        "Output: tuple of `(scalar 0-D tensor, (B,) tensor)`.\n\n"
        "The visualization renders an originals row and a corrupted-"
        "reconstructions row side by side, with the per-sample MSE annotated "
        "under each pair — you can see which samples the 'decoder' got "
        "right vs wrong."
    ),
    "stub": (
        "def ex1_mse_losses(original: Tensor, decoded: Tensor) -> tuple[Tensor, Tensor]:\n"
        '    """Return (scalar mean MSE, per-sample MSE of shape (B,))."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn.functional as F\n"
        "\n"
        "# Identity case — decoded == original → all losses zero.\n"
        "rng = t.Generator().manual_seed(0)\n"
        "orig = t.rand(4, 1, 28, 28, generator=rng)\n"
        "scalar, per_sample = ex1_mse_losses(orig, orig.clone())\n"
        "assert scalar.shape == (), f'expected scalar (), got {tuple(scalar.shape)}'\n"
        "assert per_sample.shape == (4,), f'expected (4,), got {tuple(per_sample.shape)}'\n"
        "assert t.allclose(scalar, t.tensor(0.0), atol=1e-6), f'identity scalar must be 0, got {scalar.item()}'\n"
        "assert t.allclose(per_sample, t.zeros(4), atol=1e-6), f'identity per-sample must be all 0, got {per_sample}'\n"
        "\n"
        "# Constant-offset case — each sample shifted by a known amount.\n"
        "orig2 = t.zeros(3, 1, 28, 28)\n"
        "dec2 = t.stack([t.full((1, 28, 28), v) for v in [0.0, 0.5, 1.0]], dim=0)\n"
        "scalar2, per2 = ex1_mse_losses(orig2, dec2)\n"
        "expected_per = t.tensor([0.0, 0.25, 1.0])\n"
        "assert t.allclose(per2, expected_per, atol=1e-5), f'expected {expected_per}, got {per2}'\n"
        "expected_scalar = expected_per.mean()\n"
        "assert t.allclose(scalar2, expected_scalar, atol=1e-5), 'scalar must equal mean of per-sample'\n"
        "\n"
        "# Symmetric argument check — F.mse_loss is symmetric in inputs.\n"
        "scalar_a, _ = ex1_mse_losses(orig2, dec2)\n"
        "scalar_b, _ = ex1_mse_losses(dec2, orig2)\n"
        "assert t.allclose(scalar_a, scalar_b, atol=1e-6), 'mse_loss must be symmetric'\n"
        "\n"
        "# --- Visualization: originals vs corrupted reconstructions ---\n"
        "B_viz = 6\n"
        "rng = t.Generator().manual_seed(42)\n"
        "viz_orig = t.rand(B_viz, 1, 28, 28, generator=rng)\n"
        "# Corrupt each sample with a per-sample noise scale — last samples get more noise.\n"
        "noise_scale = t.linspace(0.0, 0.4, B_viz).view(B_viz, 1, 1, 1)\n"
        "viz_dec = (viz_orig + noise_scale * t.randn(B_viz, 1, 28, 28, generator=rng)).clamp(0, 1)\n"
        "_, viz_per = ex1_mse_losses(viz_orig, viz_dec)\n"
        "fig, axes = plt.subplots(2, B_viz, figsize=(1.5 * B_viz, 3))\n"
        "for i in range(B_viz):\n"
        "    axes[0, i].imshow(viz_orig[i, 0].numpy(), cmap='gray', vmin=0, vmax=1)\n"
        "    axes[0, i].set_title('orig' if i == 0 else '')\n"
        "    axes[0, i].axis('off')\n"
        "    axes[1, i].imshow(viz_dec[i, 0].numpy(), cmap='gray', vmin=0, vmax=1)\n"
        "    axes[1, i].set_title(f'mse={viz_per[i].item():.3f}', fontsize=8)\n"
        "    axes[1, i].axis('off')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_mse_losses(original: Tensor, decoded: Tensor) -> tuple[Tensor, Tensor]:\n"
        "    import torch.nn.functional as F\n"
        "    scalar = F.mse_loss(decoded, original)                          # reduction='mean' default\n"
        "    per_elem = F.mse_loss(decoded, original, reduction='none')      # (B, 1, 28, 28)\n"
        "    per_sample = per_elem.mean(dim=[1, 2, 3])                       # (B,)\n"
        "    return scalar, per_sample"
    ),
    "solution_notes": (
        "**Why `reduction='none'` then `.mean(dim=...)`.** This is the only "
        "way to get a per-sample loss without re-implementing MSE by hand. "
        "`reduction='sum'` gives one scalar; `reduction='mean'` gives one "
        "scalar; `reduction='none'` is the one that preserves shape so you "
        "can choose your own collapse axes.\n\n"
        "**Default scalar vs explicit per-sample.** The default scalar form "
        "is what you pass to `loss.backward()` — it's symmetric, "
        "differentiable, and averaged. The per-sample form is for LOGGING — "
        "you'd never `backward()` directly on a `(B,)` vector (that "
        "implicitly sums, which is `reduction='sum'` divided by `H*W` — "
        "almost certainly not what you want).\n\n"
        "**Argument order.** Convention is `F.mse_loss(input, target)` "
        "where `input` is the model output. MSE is symmetric so it doesn't "
        "matter mathematically — but matching the convention makes the "
        "code readable."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
})


# -------------------------- ex1 / holdout-data-one-per-class
SPECS.append({
    "atom_id": "holdout-data-one-per-class",
    "subtopic": "Generative: Hold-out one-per-class data",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_HOLDOUT,
    "exercise_index": 1,
    "exercise_title": "select one sample per class for the holdout gallery",
    "slug": "select-one-per-class-holdout",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["holdout", "boolean-mask", "stack", "gallery"],
    "kcs": ["mask-equals-class", "stack-per-class-firsts"],
    "lo": (
        "Apply boolean-mask selection (`labels == c`) inside a per-class loop "
        "to extract the FIRST sample of every class, then stack into a "
        "`(num_classes, *sample_shape)` holdout tensor."
    ),
    "prompt_body": (
        "Implement `ex1_one_per_class(data, labels, num_classes)`. The "
        "logging-batch construction pattern that ARENA uses for both "
        "autoencoder and VAE training:\n\n"
        "1. `data` has shape `(N, *sample_shape)`. `labels` is a 1-D `int64` "
        "tensor of length `N`, each in `[0, num_classes)`. They are in "
        "matched order.\n"
        "2. For each class `c` in `0..num_classes-1`:\n"
        "   - Build `mask = labels == c` (a `(N,)` bool tensor).\n"
        "   - Select `data[mask][0]` — the FIRST sample whose label is `c`.\n"
        "3. Stack the resulting list with `t.stack(per_class, dim=0)`.\n"
        "4. Return the `(num_classes, *sample_shape)` tensor.\n\n"
        "Assume every class appears at least once in the batch — no need to "
        "handle empty masks.\n\n"
        "Input: `(N, *)` data, `(N,)` int64 labels, int `num_classes`.\n"
        "Output: `(num_classes, *)` tensor, indexed by class.\n\n"
        "The visualization renders the gallery as a 1×K grid so you can "
        "visually verify each row holds one sample of the corresponding "
        "class."
    ),
    "stub": (
        "def ex1_one_per_class(data: Tensor, labels: Tensor, num_classes: int) -> Tensor:\n"
        '    """Return one sample per class, stacked into (num_classes, *sample_shape)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Simple tabular case — verify FIRST-of-each-class selection.\n"
        "data = t.tensor([\n"
        "    [10.0, 20.0],   # class 1\n"
        "    [30.0, 40.0],   # class 0  <- first of class 0\n"
        "    [50.0, 60.0],   # class 2\n"
        "    [70.0, 80.0],   # class 1  <- first of class 1\n"
        "    [90.0, 99.0],   # class 0  <- second of class 0, should NOT be chosen\n"
        "    [11.0, 22.0],   # class 2  <- second of class 2, should NOT be chosen\n"
        "])\n"
        "labels = t.tensor([1, 0, 2, 1, 0, 2])\n"
        "out = ex1_one_per_class(data, labels, num_classes=3)\n"
        "assert out.shape == (3, 2), f'expected (3, 2), got {tuple(out.shape)}'\n"
        "expected = t.tensor([[30.0, 40.0],   # first of class 0\n"
        "                     [10.0, 20.0],   # first of class 1\n"
        "                     [50.0, 60.0]])  # first of class 2\n"
        "assert t.allclose(out, expected), f'first-of-each-class mismatch:\\n{out}\\nvs\\n{expected}'\n"
        "\n"
        "# Image-shaped case (mimics MNIST holdout).\n"
        "rng = t.Generator().manual_seed(0)\n"
        "N, K = 50, 10\n"
        "imgs = t.rand(N, 1, 28, 28, generator=rng)\n"
        "lbl = t.arange(N) % K   # round-robin labels: 0,1,2,...,9,0,1,2,...\n"
        "holdout = ex1_one_per_class(imgs, lbl, num_classes=K)\n"
        "assert holdout.shape == (K, 1, 28, 28), f'expected (10, 1, 28, 28), got {tuple(holdout.shape)}'\n"
        "# Round-robin: holdout[c] must equal imgs[c] (the c-th sample is class c).\n"
        "for c in range(K):\n"
        "    assert t.allclose(holdout[c], imgs[c]), f'class {c}: expected imgs[{c}], got something else'\n"
        "\n"
        "# Imbalanced-label robustness check (class 0 dominates) — still must pick the first.\n"
        "imb_labels = t.tensor([0, 0, 0, 0, 1, 0, 0, 2])\n"
        "imb_data = t.arange(8).float().unsqueeze(1)        # (8, 1) so distinct values\n"
        "imb_out = ex1_one_per_class(imb_data, imb_labels, num_classes=3)\n"
        "assert imb_out.shape == (3, 1)\n"
        "assert imb_out[0].item() == 0.0, f'first of class 0 must be data[0]=0.0, got {imb_out[0].item()}'\n"
        "assert imb_out[1].item() == 4.0, f'first of class 1 must be data[4]=4.0, got {imb_out[1].item()}'\n"
        "assert imb_out[2].item() == 7.0, f'first of class 2 must be data[7]=7.0, got {imb_out[2].item()}'\n"
        "\n"
        "# --- Visualization: 1xK gallery row ---\n"
        "fig, axes = plt.subplots(1, K, figsize=(1.2 * K, 1.5))\n"
        "for c in range(K):\n"
        "    axes[c].imshow(holdout[c, 0].numpy(), cmap='gray', vmin=0, vmax=1)\n"
        "    axes[c].set_title(f'cls {c}', fontsize=8)\n"
        "    axes[c].axis('off')\n"
        "plt.suptitle('ex1 holdout gallery — one sample per class')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_one_per_class(data: Tensor, labels: Tensor, num_classes: int) -> Tensor:\n"
        "    per_class = []\n"
        "    for c in range(num_classes):\n"
        "        mask = labels == c\n"
        "        per_class.append(data[mask][0])\n"
        "    return t.stack(per_class, dim=0)"
    ),
    "solution_notes": (
        "**`data[mask]` returns a copy.** Boolean-mask indexing always "
        "allocates fresh storage — there's no view-version that does it. "
        "That's fine here (holdout is tiny), but if you ever boolean-mask "
        "inside a tight inner loop you'll spend time profiling allocator "
        "pressure.\n\n"
        "**Why `data[mask][0]` not `data[mask][:1]`.** Both work. `[0]` "
        "returns the sample with the leading batch axis removed (shape "
        "`(*sample_shape,)`); `[:1]` keeps the axis. `t.stack(..., dim=0)` "
        "re-adds the leading axis regardless — but starting from the "
        "axis-removed form makes the stacking semantics symmetric across "
        "tabular and image cases.\n\n"
        "**Failure mode this would catch.** A buggy implementation that "
        "uses `data[mask].mean(dim=0)` (instead of `[0]`) returns the "
        "class CENTROID, which trains a different visualization (also "
        "useful, but not what ARENA's `HOLDOUT_DATA` represents)."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
})


# -------------------------- ex1 / randn-like-noise-source
SPECS.append({
    "atom_id": "randn-like-noise-source",
    "subtopic": "Generative: randn-like noise source",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_RANDN_LIKE,
    "exercise_index": 1,
    "exercise_title": "reparameterization noise that inherits dtype",
    "slug": "reparam-noise-inherits-dtype",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["randn_like", "vae", "reparameterization", "dtype"],
    "kcs": ["randn-like-preserves-dtype", "randn-like-preserves-shape"],
    "lo": (
        "Apply `t.randn_like(sigma)` (NOT `t.randn(*sigma.shape)`) to sample "
        "VAE reparameterization noise that automatically inherits the "
        "target's dtype and device."
    ),
    "prompt_body": (
        "Implement `ex1_reparameterize(mu, sigma)`. The VAE reparameterization "
        "trick — the bridge that lets gradients flow through a stochastic "
        "sampling step:\n\n"
        "1. `mu` and `sigma` both have shape `(B, latent_dim)` and matching "
        "dtype (possibly `float64` or `float16`, not just `float32`).\n"
        "2. Sample `eps` from a standard normal using `t.randn_like(sigma)` — "
        "the SHAPE, DTYPE, AND DEVICE all inherit from `sigma`. **Do NOT** "
        "use `t.randn(*sigma.shape)` (that hardcodes `float32` / CPU).\n"
        "3. Compute `z = mu + sigma * eps`.\n"
        "4. Return `z`.\n\n"
        "Input: `mu`, `sigma` — `(B, latent_dim)` float tensors (any float "
        "dtype).\n"
        "Output: `z` — `(B, latent_dim)` float tensor, same dtype as `mu` / "
        "`sigma`.\n\n"
        "The visualization plots ε samples from `randn_like` against ε "
        "samples from `randn(*shape)` to show that both produce standard-"
        "normal noise — but only `randn_like` keeps the dtype contract."
    ),
    "stub": (
        "def ex1_reparameterize(mu: Tensor, sigma: Tensor) -> Tensor:\n"
        '    """Sample z = mu + sigma * eps with eps ~ N(0, I), dtype-preserving."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Default-dtype smoke test.\n"
        "t.manual_seed(0)\n"
        "B, D = 32, 8\n"
        "mu = t.zeros(B, D)\n"
        "sigma = t.ones(B, D)\n"
        "z = ex1_reparameterize(mu, sigma)\n"
        "assert z.shape == (B, D), f'expected ({B}, {D}), got {tuple(z.shape)}'\n"
        "assert z.dtype == t.float32, f'expected float32 (from default mu/sigma), got {z.dtype}'\n"
        "# When mu=0, sigma=1 the output IS the noise — so it should be ~ N(0,1).\n"
        "assert abs(z.mean().item()) < 0.1, f'noise mean ~0, got {z.mean().item():.4f}'\n"
        "assert abs(z.std().item() - 1.0) < 0.1, f'noise std ~1, got {z.std().item():.4f}'\n"
        "\n"
        "# Dtype inheritance — this is what randn_like buys you.\n"
        "t.manual_seed(0)\n"
        "mu64 = t.zeros(4, 3, dtype=t.float64)\n"
        "sigma64 = t.ones(4, 3, dtype=t.float64)\n"
        "z64 = ex1_reparameterize(mu64, sigma64)\n"
        "assert z64.dtype == t.float64, (\n"
        "    f'dtype must inherit from sigma: expected float64, got {z64.dtype} '\n"
        "    '(likely used t.randn(*shape) instead of t.randn_like(sigma))'\n"
        ")\n"
        "\n"
        "# float16 case.\n"
        "t.manual_seed(0)\n"
        "mu16 = t.zeros(4, 3, dtype=t.float16)\n"
        "sigma16 = t.ones(4, 3, dtype=t.float16)\n"
        "z16 = ex1_reparameterize(mu16, sigma16)\n"
        "assert z16.dtype == t.float16, f'expected float16, got {z16.dtype}'\n"
        "\n"
        "# Mean shift — when sigma=0 the output equals mu exactly (no noise effect).\n"
        "mu_only = t.tensor([[5.0, -3.0, 1.5]])\n"
        "sigma_zero = t.zeros_like(mu_only)\n"
        "z_zero = ex1_reparameterize(mu_only, sigma_zero)\n"
        "assert t.allclose(z_zero, mu_only), f'sigma=0 must give z=mu exactly, got {z_zero}'\n"
        "\n"
        "# --- Visualization: randn_like vs randn(*shape) — both ~ N(0,1) ---\n"
        "t.manual_seed(0)\n"
        "N = 5000\n"
        "sigma_viz = t.ones(N)\n"
        "eps_like = t.randn_like(sigma_viz)\n"
        "eps_shape = t.randn(*sigma_viz.shape)\n"
        "fig, axes = plt.subplots(1, 2, figsize=(8, 3))\n"
        "axes[0].hist(eps_like.numpy(), bins=50, color='steelblue', edgecolor='black')\n"
        "axes[0].set_title('t.randn_like(sigma) — dtype-safe')\n"
        "axes[1].hist(eps_shape.numpy(), bins=50, color='coral', edgecolor='black')\n"
        "axes[1].set_title('t.randn(*sigma.shape) — silently float32')\n"
        "for a in axes:\n"
        "    a.set_xlabel('value'); a.set_ylabel('count'); a.grid(True, alpha=0.3)\n"
        "plt.suptitle('ex1 reparam noise — both look the same, only one preserves dtype')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_reparameterize(mu: Tensor, sigma: Tensor) -> Tensor:\n"
        "    eps = t.randn_like(sigma)\n"
        "    return mu + sigma * eps"
    ),
    "solution_notes": (
        "**`randn_like(sigma)` vs `randn(*sigma.shape)`.** Both produce "
        "standard-normal noise of the right shape. ONLY `randn_like` "
        "preserves dtype and device. The shape-call returns CPU `float32` "
        "no matter what — fine until you move the model to GPU or switch "
        "to mixed precision, at which point you get a "
        "`RuntimeError: expected all tensors to be on the same device` "
        "or silent numerical drift.\n\n"
        "**Why the noise is multiplied by sigma, not just added.** This is "
        "the heart of the reparameterization trick. `z ~ N(mu, sigma^2)` "
        "is equivalent in distribution to `mu + sigma * eps` with "
        "`eps ~ N(0, 1)`. The latter form is differentiable through `mu` "
        "and `sigma` (the stochasticity is in `eps`, which doesn't carry "
        "gradients), so the encoder learns from the decoder.\n\n"
        "**`sigma`, not `log_sigma`, in this drill.** ARENA actually has "
        "the encoder output `log_sigma` and computes `sigma = log_sigma.exp()` "
        "before this step (so sigma stays positive). We're drilling the "
        "noise-injection layer in isolation."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
})


# -------------------------- ex1 / dcgan-wrapper-netG-netD
SPECS.append({
    "atom_id": "dcgan-wrapper-netG-netD",
    "subtopic": "Generative: DCGAN netG+netD wrapper",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_DCGAN_WRAPPER,
    "exercise_index": 1,
    "exercise_title": "wrap two subnets as netG/netD with separate optimizers",
    "slug": "dcgan-wrapper-with-two-optimizers",
    "bloom_level": "Create",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["dcgan", "wrapper-module", "submodules", "two-optimizers"],
    "kcs": ["wrapper-holds-netG-netD", "wrapper-no-joint-forward"],
    "lo": (
        "Create a wrapper `nn.Module` that holds `netG` and `netD` as "
        "submodules (no joint `forward`) and verify both subnets are reachable "
        "for separate optimizer construction."
    ),
    "prompt_body": (
        "Implement `ex1_make_dcgan_wrapper(generator, discriminator)`. The "
        "DCGAN container-module pattern that ARENA's part-5 GAN training "
        "loop assumes:\n\n"
        "1. The function takes two ALREADY-CONSTRUCTED `nn.Module` "
        "instances — `generator` and `discriminator`.\n"
        "2. Return an instance of a `nn.Module` subclass that has:\n"
        "   - `wrapper.netG` set to `generator` (auto-registered as submodule).\n"
        "   - `wrapper.netD` set to `discriminator` (auto-registered as submodule).\n"
        "   - NO `forward` method — callers must invoke `wrapper.netG(noise)` "
        "or `wrapper.netD(img)` directly.\n"
        "3. `super().__init__()` must run BEFORE assigning `netG` / `netD` — "
        "otherwise the assignment raises an `AttributeError`.\n\n"
        "Input: two `nn.Module` instances.\n"
        "Output: a single `nn.Module` instance whose `.netG` and `.netD` "
        "attributes are the input modules.\n\n"
        "The visualization renders a parameter-count bar chart for both "
        "subnets to confirm the wrapper exposes them as distinct trainable "
        "submodules."
    ),
    "stub": (
        "def ex1_make_dcgan_wrapper(generator: nn.Module, discriminator: nn.Module) -> nn.Module:\n"
        '    """Build a wrapper that holds netG and netD as submodules."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "from torch import nn\n"
        "\n"
        "# Build two toy subnets.\n"
        "gen = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))\n"
        "disc = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))\n"
        "wrapper = ex1_make_dcgan_wrapper(gen, disc)\n"
        "\n"
        "# Wrapper is an nn.Module.\n"
        "assert isinstance(wrapper, nn.Module), f'expected nn.Module, got {type(wrapper)}'\n"
        "\n"
        "# netG / netD are the SAME instances we passed in (identity, not a copy).\n"
        "assert wrapper.netG is gen, 'wrapper.netG must be the SAME generator instance'\n"
        "assert wrapper.netD is disc, 'wrapper.netD must be the SAME discriminator instance'\n"
        "\n"
        "# Both subnets are auto-registered (visible via named_children).\n"
        "child_names = dict(wrapper.named_children())\n"
        "assert 'netG' in child_names, f'netG not registered as submodule; named_children={list(child_names)}'\n"
        "assert 'netD' in child_names, f'netD not registered as submodule; named_children={list(child_names)}'\n"
        "\n"
        "# Wrapper.parameters() walks BOTH subnets.\n"
        "n_params_total = sum(p.numel() for p in wrapper.parameters())\n"
        "n_params_gen = sum(p.numel() for p in gen.parameters())\n"
        "n_params_disc = sum(p.numel() for p in disc.parameters())\n"
        "assert n_params_total == n_params_gen + n_params_disc, (\n"
        "    f'wrapper params {n_params_total} != gen {n_params_gen} + disc {n_params_disc}'\n"
        ")\n"
        "\n"
        "# Construct the TWO separate optimizers the GAN loop needs.\n"
        "opt_g = t.optim.Adam(wrapper.netG.parameters(), lr=1e-3)\n"
        "opt_d = t.optim.Adam(wrapper.netD.parameters(), lr=1e-3)\n"
        "assert opt_g.param_groups[0]['params'][0] is next(gen.parameters()), 'opt_g must train netG params'\n"
        "assert opt_d.param_groups[0]['params'][0] is next(disc.parameters()), 'opt_d must train netD params'\n"
        "\n"
        "# Wrapper does NOT define a joint forward — calling wrapper(x) on a tensor of\n"
        "# arbitrary shape should fail (since base nn.Module has no forward).\n"
        "try:\n"
        "    wrapper(t.zeros(2, 8))\n"
        "    raise AssertionError('wrapper(x) must NOT work — wrapper has no joint forward')\n"
        "except (NotImplementedError, AttributeError, TypeError):\n"
        "    pass  # expected — no forward defined\n"
        "\n"
        "# Subnets work individually through the wrapper attributes.\n"
        "noise = t.randn(3, 8)\n"
        "fake = wrapper.netG(noise)\n"
        "assert fake.shape == (3, 4), f'netG output shape wrong: {tuple(fake.shape)}'\n"
        "score = wrapper.netD(fake)\n"
        "assert score.shape == (3, 1), f'netD output shape wrong: {tuple(score.shape)}'\n"
        "\n"
        "# .to(dtype=float64) on wrapper moves BOTH subnets.\n"
        "wrapper.to(t.float64)\n"
        "assert next(wrapper.netG.parameters()).dtype == t.float64, '.to() must move netG params'\n"
        "assert next(wrapper.netD.parameters()).dtype == t.float64, '.to() must move netD params'\n"
        "wrapper.to(t.float32)  # restore for downstream cells\n"
        "\n"
        "# --- Visualization: per-subnet param-count bar chart ---\n"
        "fig, ax = plt.subplots(figsize=(5, 3))\n"
        "ax.bar(['netG', 'netD'], [n_params_gen, n_params_disc],\n"
        "       color=['steelblue', 'coral'], edgecolor='black')\n"
        "ax.set_ylabel('parameter count')\n"
        "ax.set_title('ex1 wrapper exposes netG + netD as distinct trainable subnets')\n"
        "for i, n in enumerate([n_params_gen, n_params_disc]):\n"
        "    ax.text(i, n, str(n), ha='center', va='bottom')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_make_dcgan_wrapper(generator, discriminator):\n"
        "    from torch import nn\n"
        "    class DCGAN(nn.Module):\n"
        "        def __init__(self, netG, netD):\n"
        "            super().__init__()       # MUST be first — wires up _modules dict\n"
        "            self.netG = netG\n"
        "            self.netD = netD\n"
        "        # No forward method — see solution notes.\n"
        "    return DCGAN(generator, discriminator)"
    ),
    "solution_notes": (
        "**Why no joint `forward`.** A GAN's generator takes noise `(B, "
        "latent_dim)`; the discriminator takes images `(B, C, H, W)`. There "
        "isn't a single tensor signature that makes sense for both, so we "
        "don't define one. Callers always invoke `model.netG(noise)` or "
        "`model.netD(img)` explicitly.\n\n"
        "**Why a wrapper at all.** Three reasons:\n"
        "1. `model.to(device)` moves both subnets in one call (the test "
        "demonstrated this with `to(float64)`).\n"
        "2. `model.state_dict()` snapshots both for clean checkpointing.\n"
        "3. `model.train()` / `model.eval()` toggles both at once — "
        "important for BatchNorm and Dropout layers.\n\n"
        "**Two optimizers, one model.** The GAN training step alternates: "
        "update netD on a batch (with `opt_d`), then update netG on a "
        "batch (with `opt_g`). NEVER pass `model.parameters()` directly to "
        "an optimizer — that would couple the two networks into one "
        "gradient step, breaking the adversarial dynamic."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
})


# -------------------------- ex1 / broadcast-source-fanout
SPECS.append({
    "atom_id": "broadcast-source-fanout",
    "subtopic": "Generative: Broadcast source fan-out",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_BROADCAST_FANOUT,
    "exercise_index": 1,
    "exercise_title": "fan a class-embedding table to a batch via labels",
    "slug": "fanout-class-embedding-to-batch",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["embedding", "fanout", "label-indexing", "broadcast-source"],
    "kcs": ["embed-table-index-by-labels", "fanout-shape-preserves-D"],
    "lo": (
        "Apply integer-label indexing `embed_table[labels]` to fan a "
        "`(num_classes, D)` class-conditional embedding table out to a "
        "`(B, D)` per-sample embedding tensor."
    ),
    "prompt_body": (
        "Implement `ex1_fanout_class_embeddings(embed_table, labels)`. The "
        "class-conditional embedding lookup that conditional GANs and VAEs "
        "use to inject label information into the generator:\n\n"
        "1. `embed_table` has shape `(num_classes, D)` — one row per class.\n"
        "2. `labels` is a 1-D `int64` tensor of length `B`, each value in "
        "`[0, num_classes)`.\n"
        "3. Return `embed_table[labels]` — shape `(B, D)`, where row `i` is "
        "`embed_table[labels[i]]`.\n"
        "4. Two samples with the same label get the SAME embedding row "
        "(this is the 'broadcast' / fan-out semantics).\n\n"
        "Input: `embed_table` `(num_classes, D)`, `labels` `(B,)` int64.\n"
        "Output: `(B, D)` tensor of same dtype as `embed_table`.\n\n"
        "The visualization renders the embedding table on the left and the "
        "per-sample fan-out result on the right, so you can see how each "
        "row of the output is copied from the table indexed by the label."
    ),
    "stub": (
        "def ex1_fanout_class_embeddings(embed_table: Tensor, labels: Tensor) -> Tensor:\n"
        '    """Fan a (num_classes, D) embedding table out to (B, D) by labels."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Distinct embedding per class, distinct labels.\n"
        "embed = t.tensor([\n"
        "    [1.0, 2.0, 3.0],   # class 0\n"
        "    [4.0, 5.0, 6.0],   # class 1\n"
        "    [7.0, 8.0, 9.0],   # class 2\n"
        "])\n"
        "labels = t.tensor([0, 2, 1, 0, 2])  # B=5\n"
        "out = ex1_fanout_class_embeddings(embed, labels)\n"
        "assert out.shape == (5, 3), f'expected (5, 3), got {tuple(out.shape)}'\n"
        "expected = t.tensor([\n"
        "    [1.0, 2.0, 3.0],   # label 0 → row 0\n"
        "    [7.0, 8.0, 9.0],   # label 2 → row 2\n"
        "    [4.0, 5.0, 6.0],   # label 1 → row 1\n"
        "    [1.0, 2.0, 3.0],   # label 0 → row 0 (REPEATED — broadcast semantics)\n"
        "    [7.0, 8.0, 9.0],   # label 2 → row 2 (REPEATED)\n"
        "])\n"
        "assert t.allclose(out, expected), f'fan-out mismatch:\\n{out}\\nvs\\n{expected}'\n"
        "\n"
        "# All-same-label edge case — every row of output should be identical.\n"
        "same_labels = t.tensor([1, 1, 1, 1])\n"
        "out_same = ex1_fanout_class_embeddings(embed, same_labels)\n"
        "assert out_same.shape == (4, 3)\n"
        "for i in range(4):\n"
        "    assert t.allclose(out_same[i], embed[1]), 'all-same-labels must produce all-same-rows'\n"
        "\n"
        "# Shape check — output's last axis matches the table's D.\n"
        "big_embed = t.randn(10, 64, generator=t.Generator().manual_seed(0))\n"
        "big_labels = t.randint(0, 10, (32,), generator=t.Generator().manual_seed(1))\n"
        "big_out = ex1_fanout_class_embeddings(big_embed, big_labels)\n"
        "assert big_out.shape == (32, 64), f'expected (32, 64), got {tuple(big_out.shape)}'\n"
        "for i in range(32):\n"
        "    assert t.allclose(big_out[i], big_embed[big_labels[i].item()]), f'row {i} mismatch'\n"
        "\n"
        "# Dtype preserved.\n"
        "embed_f64 = embed.double()\n"
        "out_f64 = ex1_fanout_class_embeddings(embed_f64, labels)\n"
        "assert out_f64.dtype == t.float64, f'dtype must follow embed_table, got {out_f64.dtype}'\n"
        "\n"
        "# --- Visualization: table and fan-out side by side ---\n"
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))\n"
        "ax1.imshow(embed.numpy(), cmap='viridis', aspect='auto')\n"
        "ax1.set_title('embedding table (num_classes=3, D=3)')\n"
        "ax1.set_xlabel('D'); ax1.set_ylabel('class')\n"
        "ax2.imshow(out.numpy(), cmap='viridis', aspect='auto')\n"
        "ax2.set_title(f'fan-out result (B=5, D=3) — labels={labels.tolist()}')\n"
        "ax2.set_xlabel('D'); ax2.set_ylabel('batch idx')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_fanout_class_embeddings(embed_table: Tensor, labels: Tensor) -> Tensor:\n"
        "    return embed_table[labels]"
    ),
    "solution_notes": (
        "**Integer-index advanced indexing.** `embed_table[labels]` is "
        "exactly equivalent to `F.embedding(labels, embed_table)` — the "
        "Module form (`nn.Embedding`) is just a wrapper around this "
        "indexing op plus a `nn.Parameter`-wrapped weight matrix.\n\n"
        "**Fan-out IS broadcasting.** A single embedding row "
        "`embed_table[c]` gets copied to every batch position whose label "
        "equals `c`. This is the same kind of source-fan-out as "
        "`x.expand(B, D)` — one source, many destinations — but driven by "
        "label values instead of a fixed shape rule.\n\n"
        "**Gradient flow is sparse.** When `embed_table.requires_grad=True`, "
        "backward only updates the rows that were indexed. Classes absent "
        "from the batch receive zero gradient that step — which is why you "
        "see embedding tables sometimes drift slowly per row (rare classes "
        "are touched rarely)."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
})


# -------------------------- ex1 / t-stack-trajectory
SPECS.append({
    "atom_id": "t-stack-trajectory",
    "subtopic": "Generative: torch.stack trajectory",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_T_STACK_TRAJECTORY,
    "exercise_index": 1,
    "exercise_title": "stack per-step latents into a (B, T, D) trajectory",
    "slug": "stack-per-step-latents-trajectory",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["stack", "trajectory", "time-axis", "history"],
    "kcs": ["stack-new-axis-dim1", "stack-list-of-equal-shapes"],
    "lo": (
        "Apply `torch.stack(latents_list, dim=1)` to convert a length-T "
        "Python list of `(B, D)` tensors into a single `(B, T, D)` "
        "trajectory tensor with TIME as the second axis."
    ),
    "prompt_body": (
        "Implement `ex1_stack_trajectory(per_step_latents)`. The collect-"
        "history pattern that ARENA's training loops use to log per-step "
        "intermediate latents:\n\n"
        "1. `per_step_latents` is a Python list of length `T`. Each element "
        "is a `(B, D)` tensor (all the same shape).\n"
        "2. Use `t.stack(per_step_latents, dim=1)` to stack along a NEW "
        "axis at position 1.\n"
        "3. The result has shape `(B, T, D)` — batch first, then time, then "
        "feature.\n"
        "4. **DO NOT** use `dim=0` (that would give `(T, B, D)`) and **DO "
        "NOT** use `t.cat` (that would concatenate, not stack).\n\n"
        "Input: list of `T` tensors each shape `(B, D)`.\n"
        "Output: `(B, T, D)` tensor.\n\n"
        "The visualization picks the first batch element and plots each of "
        "its D latent dimensions as a separate line across T — you should "
        "see T-step trajectories per dimension."
    ),
    "stub": (
        "def ex1_stack_trajectory(per_step_latents: list[Tensor]) -> Tensor:\n"
        '    """Stack a list of (B, D) tensors into a (B, T, D) trajectory."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Tiny exact case.\n"
        "step0 = t.tensor([[1.0, 2.0], [3.0, 4.0]])   # (B=2, D=2)\n"
        "step1 = t.tensor([[5.0, 6.0], [7.0, 8.0]])\n"
        "step2 = t.tensor([[9.0, 10.], [11., 12.]])\n"
        "out = ex1_stack_trajectory([step0, step1, step2])\n"
        "assert out.shape == (2, 3, 2), f'expected (B=2, T=3, D=2), got {tuple(out.shape)}'\n"
        "# Batch element 0's trajectory.\n"
        "expected_b0 = t.tensor([[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]])\n"
        "assert t.allclose(out[0], expected_b0), f'batch 0 trajectory mismatch:\\n{out[0]}'\n"
        "# Batch element 1's trajectory.\n"
        "expected_b1 = t.tensor([[3.0, 4.0], [7.0, 8.0], [11.0, 12.0]])\n"
        "assert t.allclose(out[1], expected_b1), f'batch 1 trajectory mismatch:\\n{out[1]}'\n"
        "\n"
        "# Wrong-axis catch — make sure user used dim=1, not dim=0.\n"
        "# Output[B, T, D] = step_T[B, D]. If they did dim=0 they'd get shape (T, B, D)\n"
        "# and the (T, B, D)[0] slice would equal step0. Detect via shape mismatch.\n"
        "assert out[0].shape == (3, 2), (\n"
        "    f'out[0].shape={tuple(out[0].shape)} — expected (T=3, D=2). '\n"
        "    'If you got (B=2, D=2) you stacked along dim=0 instead of dim=1.'\n"
        ")\n"
        "\n"
        "# Realistic shape.\n"
        "rng = t.Generator().manual_seed(0)\n"
        "B, D, T_steps = 8, 16, 20\n"
        "history = [t.randn(B, D, generator=rng) for _ in range(T_steps)]\n"
        "traj = ex1_stack_trajectory(history)\n"
        "assert traj.shape == (B, T_steps, D), f'expected ({B},{T_steps},{D}), got {tuple(traj.shape)}'\n"
        "# Spot-check several (b, t, d) cells.\n"
        "for b in [0, 3, 7]:\n"
        "    for tt in [0, 5, 19]:\n"
        "        for d in [0, 8, 15]:\n"
        "            assert traj[b, tt, d] == history[tt][b, d], f'cell ({b},{tt},{d}) mismatch'\n"
        "\n"
        "# Single-step degenerate — T=1 should give shape (B, 1, D).\n"
        "one_step = [t.randn(4, 5, generator=t.Generator().manual_seed(0))]\n"
        "one_traj = ex1_stack_trajectory(one_step)\n"
        "assert one_traj.shape == (4, 1, 5), f'T=1 must give (B, 1, D), got {tuple(one_traj.shape)}'\n"
        "\n"
        "# --- Visualization: per-dimension trajectory of batch element 0 ---\n"
        "fig, ax = plt.subplots(figsize=(7, 3))\n"
        "for d in range(D):\n"
        "    ax.plot(range(T_steps), traj[0, :, d].numpy(), alpha=0.6)\n"
        "ax.set_xlabel('step (t)')\n"
        "ax.set_ylabel('latent value')\n"
        "ax.set_title(f'ex1 batch-element-0 trajectory across {T_steps} steps ({D} latent dims)')\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_stack_trajectory(per_step_latents: list[Tensor]) -> Tensor:\n"
        "    return t.stack(per_step_latents, dim=1)"
    ),
    "solution_notes": (
        "**Why `dim=1`, not `dim=0`.** Both are technically correct stacks — "
        "they just disagree on which axis becomes time. `dim=1` gives "
        "`(B, T, D)`, which matches the `(batch, sequence, features)` "
        "convention used by every recurrent / sequence module in PyTorch. "
        "`dim=0` gives `(T, B, D)`, which is what some older recurrence "
        "code expects but is otherwise idiosyncratic.\n\n"
        "**Why `stack`, not `cat`.** `stack` introduces a NEW axis at "
        "`dim`. `cat` concatenates along an EXISTING axis. If you `cat` a "
        "list of `(B, D)` along `dim=1`, you get `(B, T*D)` — the time "
        "and feature axes get fused, which is almost never what you want.\n\n"
        "**Shape rigidity.** Every tensor in the list must have the SAME "
        "shape (otherwise `stack` raises). A common bug: appending a "
        "`(B, D)` and a `(D,)` from the same loop (e.g. by indexing with "
        "`[0]` instead of `[:1]` for batch=1) — `stack` will fail with a "
        "fairly readable error."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
})


# -------------------------- ex1 / requires-grad-leaf-assert
SPECS.append({
    "atom_id": "requires-grad-leaf-assert",
    "subtopic": "Generative: requires_grad leaf assert",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_REQ_GRAD_LEAF,
    "exercise_index": 1,
    "exercise_title": "guard optimizer init with leaf + requires_grad asserts",
    "slug": "guard-optimizer-with-leaf-asserts",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["leaf", "requires_grad", "optimizer", "defensive-assert"],
    "kcs": ["assert-p-is-leaf", "assert-p-requires-grad"],
    "lo": (
        "Apply `assert p.is_leaf and p.requires_grad` defensively over an "
        "iterable of would-be optimizer params, raising on the FIRST "
        "non-leaf or grad-disabled tensor with a useful message."
    ),
    "prompt_body": (
        "Implement `ex1_assert_optim_ready(params)`. The safety check that "
        "saves you from silent-no-op optimizer bugs:\n\n"
        "1. `params` is an iterable of tensors that you intend to pass to "
        "an optimizer constructor.\n"
        "2. For each tensor `p`:\n"
        "   - Assert `p.is_leaf` — non-leaf tensors are op outputs, not "
        "trainable variables; `opt.step()` silently skips them.\n"
        "   - Assert `p.requires_grad` — params with `requires_grad=False` "
        "never receive gradient, so the optimizer can't update them.\n"
        "3. If a param fails EITHER assertion, raise `AssertionError` with a "
        "message that names which check failed and includes the tensor "
        "shape (so the user can find the offender).\n"
        "4. If all params pass, return `True`.\n\n"
        "Input: list of `Tensor` (possibly `nn.Parameter`).\n"
        "Output: `True` if all checks pass; otherwise raise `AssertionError`.\n\n"
        "The visualization shows a pass/fail grid across a mix of leaf / "
        "non-leaf / grad-on / grad-off params so you can see which categories "
        "the optimizer would silently skip."
    ),
    "stub": (
        "def ex1_assert_optim_ready(params) -> bool:\n"
        '    """Assert every param is a grad-tracking leaf, suitable for an optimizer."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "from torch import nn\n"
        "\n"
        "# All-pass case — fresh nn.Parameter list.\n"
        "good = [nn.Parameter(t.randn(3)), nn.Parameter(t.randn(2, 2))]\n"
        "assert ex1_assert_optim_ready(good) is True, 'all-leaf, requires_grad=True should pass'\n"
        "\n"
        "# requires_grad=False case → must raise with a useful message.\n"
        "frozen = t.randn(3)               # requires_grad=False by default for raw tensor\n"
        "try:\n"
        "    ex1_assert_optim_ready([frozen])\n"
        "    raise AssertionError('non-grad tensor must trigger the assertion')\n"
        "except AssertionError as e:\n"
        "    msg = str(e)\n"
        "    assert ('requires_grad' in msg or 'grad' in msg.lower()), (\n"
        "        f'error must mention requires_grad / grad — got: {msg!r}'\n"
        "    )\n"
        "\n"
        "# Non-leaf case — produce a non-leaf tensor by doing an op on a leaf.\n"
        "leaf_param = nn.Parameter(t.randn(4))\n"
        "non_leaf = leaf_param * 2   # non-leaf (result of multiplication)\n"
        "assert not non_leaf.is_leaf, 'sanity: non_leaf must indeed be non-leaf'\n"
        "assert non_leaf.requires_grad, 'sanity: non_leaf must propagate requires_grad'\n"
        "try:\n"
        "    ex1_assert_optim_ready([non_leaf])\n"
        "    raise AssertionError('non-leaf tensor must trigger the assertion')\n"
        "except AssertionError as e:\n"
        "    msg = str(e)\n"
        "    assert ('leaf' in msg.lower() or 'is_leaf' in msg), (\n"
        "        f'error must mention leaf / is_leaf — got: {msg!r}'\n"
        "    )\n"
        "\n"
        "# Mixed-failure case — the FIRST bad param triggers, downstream params do not run.\n"
        "ok = nn.Parameter(t.randn(3))\n"
        "bad = t.randn(2)  # requires_grad=False\n"
        "trigger_count = {'n': 0}\n"
        "class CountingParam:\n"
        "    # Lightweight wrapper that simulates a third 'param' we should never reach.\n"
        "    def __init__(self):\n"
        "        self._real = nn.Parameter(t.randn(1))\n"
        "    @property\n"
        "    def is_leaf(self):\n"
        "        trigger_count['n'] += 1\n"
        "        return self._real.is_leaf\n"
        "    @property\n"
        "    def requires_grad(self):\n"
        "        return self._real.requires_grad\n"
        "    @property\n"
        "    def shape(self):\n"
        "        return self._real.shape\n"
        "never_reached = CountingParam()\n"
        "try:\n"
        "    ex1_assert_optim_ready([ok, bad, never_reached])\n"
        "    raise AssertionError('mixed list with bad param must raise')\n"
        "except AssertionError:\n"
        "    pass\n"
        "# We don't strictly require short-circuit, but it's a hint about implementation.\n"
        "\n"
        "# Empty-iterable edge case — vacuously true.\n"
        "assert ex1_assert_optim_ready([]) is True, 'empty params list is vacuously fine'\n"
        "\n"
        "# Real Module case — model.parameters() should all pass.\n"
        "model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))\n"
        "assert ex1_assert_optim_ready(list(model.parameters())) is True, 'fresh Module params must pass'\n"
        "\n"
        "# --- Visualization: 2x2 pass/fail grid over leaf/non-leaf × grad-on/grad-off ---\n"
        "cases = {\n"
        "    'leaf + grad-on':    nn.Parameter(t.randn(3)),\n"
        "    'leaf + grad-off':   t.randn(3),                                            # requires_grad=False\n"
        "    'non-leaf + grad-on':  nn.Parameter(t.randn(3)) * 2,                        # op output\n"
        "    'non-leaf + grad-off': (t.randn(3) * 2).detach(),                           # detached\n"
        "}\n"
        "results = []\n"
        "for label, p in cases.items():\n"
        "    try:\n"
        "        ex1_assert_optim_ready([p])\n"
        "        results.append((label, True))\n"
        "    except AssertionError:\n"
        "        results.append((label, False))\n"
        "fig, ax = plt.subplots(figsize=(6, 3))\n"
        "labels_v = [r[0] for r in results]\n"
        "colors = ['#3a8' if r[1] else '#d33' for r in results]\n"
        "ax.barh(labels_v, [1] * len(results), color=colors, edgecolor='black')\n"
        "for i, (lab, ok) in enumerate(results):\n"
        "    ax.text(0.5, i, 'PASS' if ok else 'FAIL', ha='center', va='center', color='white', weight='bold')\n"
        "ax.set_xlim(0, 1)\n"
        "ax.set_xticks([])\n"
        "ax.set_title('ex1 leaf + requires_grad guard — only top case is optimizer-safe')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex1_assert_optim_ready(params) -> bool:\n"
        "    for p in params:\n"
        "        assert p.is_leaf, (\n"
        "            f'param shape={tuple(p.shape)} is NON-LEAF — optimizer will silently skip it. '\n"
        "            'Move the whole nn.Module with .to(device), not individual Parameters.'\n"
        "        )\n"
        "        assert p.requires_grad, (\n"
        "            f'param shape={tuple(p.shape)} has requires_grad=False — '\n"
        "            'no gradient will ever flow, optimizer cannot update it.'\n"
        "        )\n"
        "    return True"
    ),
    "solution_notes": (
        "**Why this saves hours.** A common bug: you `.to(device)` a "
        "single `nn.Parameter` instead of the whole Module. The result is "
        "a NEW non-leaf tensor that's `requires_grad=True` and shares no "
        "memory with the original. The optimizer accepts it, `.step()` "
        "runs without error, and silently does nothing. Loss plateaus, "
        "you debug for a day. This assert catches it at construction.\n\n"
        "**`is_leaf` semantics.** A tensor is a leaf if it has no "
        "`grad_fn`. Leaves are: (a) tensors you constructed via "
        "`nn.Parameter(...)` or `t.tensor(..., requires_grad=True)`, and "
        "(b) any tensor with `requires_grad=False`. Non-leaves are op "
        "outputs that participate in autograd — they get gradients during "
        "backward (visible at `.grad` only if `.retain_grad()`), but "
        "`opt.step()` only updates leaves.\n\n"
        "**Why two separate asserts.** `requires_grad=True` does NOT "
        "imply `is_leaf=True`. A non-leaf can have grad enabled (it "
        "almost always does — that's how autograd flows). Both conditions "
        "must hold for the optimizer to do useful work."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
})


# ---------------------------------------------------------------- emit

for spec in SPECS:
    path = emit_standalone(spec)
    rel = path.relative_to(path.parents[3])
    print(f"wrote {rel}")
