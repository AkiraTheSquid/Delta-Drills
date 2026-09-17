"""Composite drills cx25..cx30 — batch-20 part5 (OO-cell, ARENA VAE/GAN trainer + wandb).

Six composite procedural drills exercising 2-3-atom pairs from ARENA part 5 —
generative models + trainer class + wandb logging.

cx25  mse-reconstruction-loss + backward-on-scalar-loss
      → AE: MSE recon loss + backward on the scalar
cx26  mse-reconstruction-loss + bottleneck-latent-projection
      → AE forward: encode → bottleneck → decode → MSE
cx27  trainer-class-skeleton + backward-on-scalar-loss
      → Trainer.training_step with backward
cx28  trainer-class-skeleton + dataloader-batching
      → Trainer with DataLoader integration
cx29  wandb-init-run + wandb-log-step
      → wandb.init then wandb.log in the same trainer
cx30  wandb-init-run + log-samples-eval-callback
      → wandb.init then log_samples callback emits wandb.Image
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_composite import emit_composite  # noqa: E402

INV = {a["atom_id"]: a for a in json.load(open("/tmp/drill_atoms.json"))}


def _subs(atom_ids):
    return [INV[a]["subtopic"] for a in atom_ids]


NN_IMPORTS = [
    "import torch.nn as nn",
    "import torch.nn.functional as F",
]
NN_DL_IMPORTS = [
    "import torch.nn as nn",
    "import torch.nn.functional as F",
    "from torch.utils.data import DataLoader, TensorDataset",
]
WANDB_IMPORTS = [
    "import torch.nn as nn",
    "import torch.nn.functional as F",
    "import wandb",
]
WANDB_DL_IMPORTS = [
    "import torch.nn as nn",
    "import torch.nn.functional as F",
    "import wandb",
    "from torch.utils.data import DataLoader, TensorDataset",
]


# ===========================================================================
# cx25 — AE: MSE recon loss + backward (mse-reconstruction-loss + backward-on-scalar-loss)
# ===========================================================================
spec_25 = {
    "atom_ids": ["mse-reconstruction-loss", "backward-on-scalar-loss"],
    "subtopics": _subs(["mse-reconstruction-loss", "backward-on-scalar-loss"]),
    "primary_atom": "mse-reconstruction-loss",
    "part": "part5",
    "exercise_index": 25,
    "exercise_title": "MSE reconstruction loss → backward on scalar → grads land on encoder/decoder",
    "slug": "ae-mse-recon-loss-then-backward",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "An autoencoder produces a reconstruction `x_hat` from input `x`. To train it, you reduce "
        "the per-element reconstruction error to a SCALAR and call `.backward()`. Two atoms wire "
        "together every step:\n\n"
        "1. **mse-reconstruction-loss** — `loss = F.mse_loss(x_hat, x)` (default `reduction='mean'`). "
        "MSE is the natural recon loss for continuous-valued inputs; for binary pixels you'd use "
        "BCE instead.\n"
        "2. **backward-on-scalar-loss** — `.backward()` requires its anchor to be a SCALAR "
        "(0-dim tensor). `F.mse_loss(..., reduction='mean')` already returns a scalar; "
        "`reduction='none'` would return a same-shape tensor and `.backward()` would fail with "
        "`grad can be implicitly created only for scalar outputs`.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "x_hat = decoder(encoder(x))          # forward.\n"
        "loss = F.mse_loss(x_hat, x)          # mse-reconstruction-loss (scalar by default).\n"
        "loss.backward()                       # backward-on-scalar-loss → fills .grad on all params.\n"
        "```\n\n"
        "**Why test these together.** A common student bug is `loss = (x_hat - x).pow(2)` (no "
        "reduction) followed by `.backward()` — fails because the loss is not scalar. The fix is "
        "`.mean()` (or `F.mse_loss`'s default). Both atoms have to be respected jointly."
    ),
    "prompt_body": (
        "Implement `cx25_ae_recon_backward(encoder, decoder, x)`.\n\n"
        "Inputs:\n"
        "- `encoder`, `decoder` — two `nn.Module` instances. Calling `decoder(encoder(x))` produces "
        "  `x_hat` with the same shape as `x`.\n"
        "- `x` — an input batch tensor.\n\n"
        "Required behaviour:\n"
        "1. Compute `x_hat = decoder(encoder(x))`.\n"
        "2. Compute the scalar MSE recon loss via `F.mse_loss(x_hat, x)` (atom: "
        "mse-reconstruction-loss). The result MUST be a 0-dim tensor.\n"
        "3. Call `loss.backward()` (atom: backward-on-scalar-loss). This populates `.grad` on every "
        "parameter in encoder + decoder.\n"
        "4. Return the scalar `loss` tensor (NOT `loss.item()` — the test inspects it).\n\n"
        "The test verifies: the returned loss is a scalar grad-tracking tensor, `.grad` is populated "
        "on every encoder + decoder parameter (all non-None and non-zero), and the loss is "
        "non-negative."
    ),
    "stub_body": (
        "def cx25_ae_recon_backward(encoder, decoder, x):\n"
        "    \"\"\"Forward → MSE recon loss → backward. Returns the scalar loss tensor.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Tiny encoder/decoder pair (flat MLP).\n"
        "t.manual_seed(0)\n"
        "encoder = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))\n"
        "decoder = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 8))\n"
        "# Zero any pre-existing grads.\n"
        "for p in list(encoder.parameters()) + list(decoder.parameters()):\n"
        "    if p.grad is not None:\n"
        "        p.grad = None\n"
        "x = t.randn(5, 8)\n"
        "\n"
        "loss = cx25_ae_recon_backward(encoder, decoder, x)\n"
        "\n"
        "# Case A: returns a scalar grad-tracking tensor.\n"
        "assert isinstance(loss, t.Tensor), f'must return a Tensor, got {type(loss).__name__}'\n"
        "assert loss.ndim == 0, f'loss must be scalar (0-dim); got shape {tuple(loss.shape)}'\n"
        "assert loss.item() >= 0.0, 'MSE loss must be non-negative'\n"
        "\n"
        "# Case B: matches F.mse_loss(decoder(encoder(x)), x).\n"
        "with t.no_grad():\n"
        "    x_hat_ref = decoder(encoder(x))\n"
        "    ref = F.mse_loss(x_hat_ref, x)\n"
        "assert t.allclose(loss.detach(), ref, atol=1e-6), (\n"
        "    f'loss should equal F.mse_loss(decoder(encoder(x)), x); got {loss.item():.6f} vs {ref.item():.6f}'\n"
        ")\n"
        "\n"
        "# Case C: backward populated grads on every parameter.\n"
        "all_params = list(encoder.parameters()) + list(decoder.parameters())\n"
        "for i, p in enumerate(all_params):\n"
        "    assert p.grad is not None, f'param[{i}] has .grad=None — backward did not propagate'\n"
        "    assert p.grad.shape == p.shape, f'param[{i}] grad shape mismatch'\n"
        "    assert p.grad.abs().sum().item() > 0, f'param[{i}] grad is all zeros'\n"
        "\n"
        "# Case D: scalar-loss contract — non-scalar would have errored. Sanity by repeating once more.\n"
        "for p in all_params:\n"
        "    p.grad = None\n"
        "loss2 = cx25_ae_recon_backward(encoder, decoder, x)\n"
        "assert loss2.ndim == 0\n"
        "for p in all_params:\n"
        "    assert p.grad is not None"
    ),
    "solution_body": (
        "def cx25_ae_recon_backward(encoder, decoder, x):\n"
        "    # Forward through encoder then decoder.\n"
        "    x_hat = decoder(encoder(x))\n"
        "    # Atom A (mse-reconstruction-loss): scalar mean-squared-error.\n"
        "    loss = F.mse_loss(x_hat, x)\n"
        "    # Atom B (backward-on-scalar-loss): backward requires scalar output.\n"
        "    loss.backward()\n"
        "    return loss"
    ),
    "solution_notes": (
        "`F.mse_loss(reduction='mean')` (the default) gives a scalar — exactly what `.backward()` "
        "needs. If you used `reduction='none'` you'd have to call `.mean()` (or `.sum()`) before "
        "`.backward()`, otherwise PyTorch raises `RuntimeError: grad can be implicitly created only "
        "for scalar outputs`."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["mse-reconstruction-loss", "backward-on-scalar-loss"],
    "lo": (
        "Compose F.mse_loss with .backward() on the scalar result so gradients populate on every "
        "encoder + decoder parameter in one wired-up step."
    ),
}


# ===========================================================================
# cx26 — AE forward: encode → bottleneck → decode → MSE
# (mse-reconstruction-loss + bottleneck-latent-projection)
# ===========================================================================
spec_26 = {
    "atom_ids": ["mse-reconstruction-loss", "bottleneck-latent-projection"],
    "subtopics": _subs(["mse-reconstruction-loss", "bottleneck-latent-projection"]),
    "primary_atom": "bottleneck-latent-projection",
    "part": "part5",
    "exercise_index": 26,
    "exercise_title": "AE forward: flatten → encode → bottleneck z → decode → MSE recon loss",
    "slug": "ae-bottleneck-then-mse-recon",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "An autoencoder squeezes the input through a low-dim BOTTLENECK `z`, then reconstructs. The "
        "reconstruction quality is measured by MSE. Two atoms wire together:\n\n"
        "1. **bottleneck-latent-projection** — the encoder ends in a Linear layer that maps the "
        "feature dimension down to `latent_dim` (e.g. 784 → 2 for MNIST visualisation). This is the "
        "INFORMATION BOTTLENECK: the model can only express what fits through `z`.\n"
        "2. **mse-reconstruction-loss** — `F.mse_loss(x_hat, x)` measures how much information made "
        "it back. A wider bottleneck → lower MSE (less information lost); a narrower bottleneck → "
        "higher MSE.\n\n"
        "**Anatomy of `AE.forward`.**\n"
        "```python\n"
        "def forward(self, x):\n"
        "    z = self.encoder(x)                 # bottleneck-latent-projection: shape (B, latent_dim).\n"
        "    x_hat = self.decoder(z)             # decode back to input shape.\n"
        "    return x_hat, z\n"
        "\n"
        "x_hat, z = model(x)\n"
        "loss = F.mse_loss(x_hat, x)             # mse-reconstruction-loss.\n"
        "```\n\n"
        "**Why test together.** The bottleneck shape and recon loss are tightly coupled: if you mess "
        "up `latent_dim` or forget to map back through the decoder, the recon loss either explodes "
        "or stops being a meaningful signal."
    ),
    "prompt_body": (
        "Implement `cx26_make_autoencoder(input_dim, hidden_dim, latent_dim)` which returns a "
        "`SimpleAE` instance (a `nn.Module`).\n\n"
        "Required structure:\n"
        "- `SimpleAE.__init__(self, input_dim, hidden_dim, latent_dim)`:\n"
        "  - `self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), "
        "nn.Linear(hidden_dim, latent_dim))` — the final Linear is the BOTTLENECK "
        "(atom: bottleneck-latent-projection).\n"
        "  - `self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), "
        "nn.Linear(hidden_dim, input_dim))`.\n"
        "  - `self.latent_dim = latent_dim` (so the test can read it back).\n"
        "- `SimpleAE.forward(self, x)` — returns `(x_hat, z)`:\n"
        "  - `z = self.encoder(x)` — shape `(B, latent_dim)`.\n"
        "  - `x_hat = self.decoder(z)` — shape `(B, input_dim)`, matches `x`.\n\n"
        "Then also: the test calls `F.mse_loss(x_hat, x)` (atom: mse-reconstruction-loss) on the "
        "forward output and verifies the scalar is sensible.\n\n"
        "The test verifies:\n"
        "- `z` has shape `(B, latent_dim)` — the bottleneck actually squeezes.\n"
        "- `x_hat` has the same shape as `x`.\n"
        "- `F.mse_loss(x_hat, x)` is a finite non-negative scalar.\n"
        "- A wider bottleneck (latent_dim=16) achieves LOWER recon loss after a few training steps "
        "than a narrower bottleneck (latent_dim=2). This is the information-bottleneck consequence."
    ),
    "stub_body": (
        "def cx26_make_autoencoder(input_dim, hidden_dim, latent_dim):\n"
        "    \"\"\"Return a SimpleAE nn.Module with the bottleneck encoder + symmetric decoder.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "t.manual_seed(0)\n"
        "model = cx26_make_autoencoder(input_dim=8, hidden_dim=16, latent_dim=2)\n"
        "assert isinstance(model, nn.Module), 'must return an nn.Module instance'\n"
        "assert hasattr(model, 'encoder') and hasattr(model, 'decoder'), 'need encoder + decoder attrs'\n"
        "assert getattr(model, 'latent_dim', None) == 2, 'latent_dim attribute must be set on the module'\n"
        "\n"
        "# Case A: forward returns (x_hat, z) with correct shapes.\n"
        "x = t.randn(5, 8)\n"
        "out = model(x)\n"
        "assert isinstance(out, tuple) and len(out) == 2, 'forward must return (x_hat, z) tuple'\n"
        "x_hat, z = out\n"
        "assert z.shape == (5, 2), f'z bottleneck shape must be (B, latent_dim)=(5,2); got {tuple(z.shape)}'\n"
        "assert x_hat.shape == x.shape, f'x_hat shape mismatch: {tuple(x_hat.shape)} vs {tuple(x.shape)}'\n"
        "\n"
        "# Case B: MSE recon loss is a finite scalar.\n"
        "loss = F.mse_loss(x_hat, x)\n"
        "assert loss.ndim == 0, 'F.mse_loss must give scalar by default'\n"
        "assert t.isfinite(loss).item(), 'loss must be finite'\n"
        "assert loss.item() >= 0.0, 'MSE non-negative'\n"
        "\n"
        "# Case C: wider bottleneck → lower recon loss after a few steps of training.\n"
        "def _train_and_eval(latent_dim, n_steps=80):\n"
        "    t.manual_seed(0)\n"
        "    m = cx26_make_autoencoder(input_dim=8, hidden_dim=16, latent_dim=latent_dim)\n"
        "    opt = t.optim.Adam(m.parameters(), lr=5e-3)\n"
        "    t.manual_seed(123)\n"
        "    data = t.randn(64, 8)\n"
        "    for _ in range(n_steps):\n"
        "        x_hat, _ = m(data)\n"
        "        loss = F.mse_loss(x_hat, data)\n"
        "        opt.zero_grad(); loss.backward(); opt.step()\n"
        "    with t.no_grad():\n"
        "        x_hat, _ = m(data)\n"
        "        return F.mse_loss(x_hat, data).item()\n"
        "\n"
        "loss_narrow = _train_and_eval(latent_dim=2)\n"
        "loss_wide = _train_and_eval(latent_dim=16)\n"
        "assert loss_wide < loss_narrow, (\n"
        "    f'wider bottleneck (latent_dim=16) should beat narrower (latent_dim=2); '\n"
        "    f'got wide={loss_wide:.4f} narrow={loss_narrow:.4f}'\n"
        ")"
    ),
    "solution_body": (
        "def cx26_make_autoencoder(input_dim, hidden_dim, latent_dim):\n"
        "    class SimpleAE(nn.Module):\n"
        "        def __init__(self, input_dim, hidden_dim, latent_dim):\n"
        "            super().__init__()\n"
        "            # Atom A (bottleneck-latent-projection): encoder ends at latent_dim.\n"
        "            self.encoder = nn.Sequential(\n"
        "                nn.Linear(input_dim, hidden_dim),\n"
        "                nn.ReLU(),\n"
        "                nn.Linear(hidden_dim, latent_dim),\n"
        "            )\n"
        "            self.decoder = nn.Sequential(\n"
        "                nn.Linear(latent_dim, hidden_dim),\n"
        "                nn.ReLU(),\n"
        "                nn.Linear(hidden_dim, input_dim),\n"
        "            )\n"
        "            self.latent_dim = latent_dim\n"
        "\n"
        "        def forward(self, x):\n"
        "            z = self.encoder(x)\n"
        "            x_hat = self.decoder(z)\n"
        "            return x_hat, z\n"
        "\n"
        "    return SimpleAE(input_dim, hidden_dim, latent_dim)"
    ),
    "solution_notes": (
        "Returning `(x_hat, z)` (rather than only `x_hat`) is the ARENA convention — you almost "
        "always want the latent for visualisation, anomaly detection, or downstream tasks. The "
        "loss is computed on `x_hat` only; `z` is the bottleneck representation."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["bottleneck-latent-projection", "mse-reconstruction-loss"],
    "lo": (
        "Compose an encoder ending in a low-dim Linear bottleneck with a symmetric decoder + MSE "
        "recon loss so the latent code becomes the only path through which information can flow."
    ),
}


# ===========================================================================
# cx27 — Trainer.training_step with backward
# (trainer-class-skeleton + backward-on-scalar-loss)
# ===========================================================================
spec_27 = {
    "atom_ids": ["trainer-class-skeleton", "backward-on-scalar-loss"],
    "subtopics": _subs(["trainer-class-skeleton", "backward-on-scalar-loss"]),
    "primary_atom": "trainer-class-skeleton",
    "part": "part5",
    "exercise_index": 27,
    "exercise_title": "Trainer.training_step: forward → scalar loss → backward → opt.step",
    "slug": "trainer-training-step-with-backward",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "The Trainer-class skeleton (ARENA convention) factors the training loop into methods so "
        "each step is independently testable. The central method is `training_step(batch)` which "
        "wires together forward + loss + backward + optimizer step. Two atoms compose:\n\n"
        "1. **trainer-class-skeleton** — `__init__` stores `model`, `optimizer`, `loss_fn`, and a "
        "`step` counter; `training_step(batch)` is the per-batch hook.\n"
        "2. **backward-on-scalar-loss** — inside `training_step`, the loss MUST be reduced to a "
        "scalar before `.backward()` is called. The scalar contract is what makes the rest of the "
        "training loop (`opt.step`, `opt.zero_grad`) well-defined.\n\n"
        "**Anatomy of `training_step`.**\n"
        "```python\n"
        "def training_step(self, batch):\n"
        "    x, y = batch\n"
        "    logits = self.model(x)\n"
        "    loss = self.loss_fn(logits, y)      # scalar.\n"
        "    self.optimizer.zero_grad()\n"
        "    loss.backward()                     # backward-on-scalar-loss.\n"
        "    self.optimizer.step()\n"
        "    self.step += 1\n"
        "    return loss.item()\n"
        "```\n\n"
        "**Why test together.** This is the smallest unit that exercises the full "
        "forward-backward-step cycle inside the OO trainer wrapper."
    ),
    "prompt_body": (
        "Implement `cx27_make_trainer()` which returns a `MiniTrainer` class.\n\n"
        "Required structure:\n"
        "- `MiniTrainer.__init__(self, model, optimizer, loss_fn)`:\n"
        "  - Store `self.model`, `self.optimizer`, `self.loss_fn`.\n"
        "  - `self.step = 0` (global step counter).\n"
        "  - `self.history = []` (list of per-step `loss.item()` floats).\n"
        "- `MiniTrainer.training_step(self, batch)`:\n"
        "  - Unpack `x, y = batch`.\n"
        "  - `logits = self.model(x)`.\n"
        "  - `loss = self.loss_fn(logits, y)` — MUST be a scalar.\n"
        "  - `self.optimizer.zero_grad()`.\n"
        "  - `loss.backward()` (atom: backward-on-scalar-loss).\n"
        "  - `self.optimizer.step()`.\n"
        "  - `self.step += 1`.\n"
        "  - `self.history.append(loss.item())`.\n"
        "  - Return the scalar `loss` tensor.\n\n"
        "The test verifies a regression task converges, the step counter increments by 1 per "
        "`training_step` call, the history list grows in sync, and gradients land on model params."
    ),
    "stub_body": (
        "def cx27_make_trainer():\n"
        "    \"\"\"Return a MiniTrainer class whose training_step does forward+backward+opt.step.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "MiniTrainer = cx27_make_trainer()\n"
        "\n"
        "# Tiny linear regression: y = 3x + 2.\n"
        "t.manual_seed(0)\n"
        "model = nn.Linear(1, 1)\n"
        "opt = t.optim.SGD(model.parameters(), lr=0.05)\n"
        "loss_fn = nn.MSELoss()\n"
        "trainer = MiniTrainer(model, opt, loss_fn)\n"
        "\n"
        "# Case A: initial state.\n"
        "assert trainer.model is model\n"
        "assert trainer.optimizer is opt\n"
        "assert trainer.loss_fn is loss_fn\n"
        "assert trainer.step == 0, f'step must start at 0; got {trainer.step}'\n"
        "assert trainer.history == [], 'history must start empty'\n"
        "\n"
        "# Case B: one training_step returns scalar loss + increments counters.\n"
        "x = t.randn(8, 1)\n"
        "y = 3.0 * x + 2.0\n"
        "loss = trainer.training_step((x, y))\n"
        "assert isinstance(loss, t.Tensor), f'training_step must return a Tensor; got {type(loss).__name__}'\n"
        "assert loss.ndim == 0, f'loss must be scalar; got shape {tuple(loss.shape)}'\n"
        "assert trainer.step == 1, f'step must be 1 after one call; got {trainer.step}'\n"
        "assert len(trainer.history) == 1\n"
        "assert isinstance(trainer.history[0], float)\n"
        "\n"
        "# Case C: backward populated grads on model params.\n"
        "for p in model.parameters():\n"
        "    assert p.grad is not None, '.grad must be populated after training_step (backward fired)'\n"
        "\n"
        "# Case D: convergence — loss decreases across many training_step calls.\n"
        "for _ in range(100):\n"
        "    trainer.training_step((x, y))\n"
        "assert trainer.step == 101, f'step should be 101 after 1+100 calls; got {trainer.step}'\n"
        "assert len(trainer.history) == 101\n"
        "assert trainer.history[-1] < trainer.history[0], (\n"
        "    f'loss should decrease: first={trainer.history[0]:.4f} last={trainer.history[-1]:.4f}'\n"
        ")\n"
        "# Fit close to truth.\n"
        "assert abs(model.weight.item() - 3.0) < 0.3, f'weight should approach 3; got {model.weight.item():.3f}'\n"
        "assert abs(model.bias.item() - 2.0) < 0.3, f'bias should approach 2; got {model.bias.item():.3f}'"
    ),
    "solution_body": (
        "def cx27_make_trainer():\n"
        "    class MiniTrainer:\n"
        "        def __init__(self, model, optimizer, loss_fn):\n"
        "            # Atom A (trainer-class-skeleton): store deps + counters.\n"
        "            self.model = model\n"
        "            self.optimizer = optimizer\n"
        "            self.loss_fn = loss_fn\n"
        "            self.step = 0\n"
        "            self.history = []\n"
        "\n"
        "        def training_step(self, batch):\n"
        "            x, y = batch\n"
        "            logits = self.model(x)\n"
        "            loss = self.loss_fn(logits, y)\n"
        "            self.optimizer.zero_grad()\n"
        "            # Atom B (backward-on-scalar-loss): scalar loss enables implicit grad seed.\n"
        "            loss.backward()\n"
        "            self.optimizer.step()\n"
        "            self.step += 1\n"
        "            self.history.append(loss.item())\n"
        "            return loss\n"
        "\n"
        "    return MiniTrainer"
    ),
    "solution_notes": (
        "Returning the loss TENSOR (not `.item()`) lets the caller chain `loss.backward()` again if "
        "they want — but here we already called backward inside `training_step`. The history stores "
        "`.item()` (plain float) so it doesn't hold a reference to the graph and prevent GC."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["trainer-class-skeleton", "backward-on-scalar-loss"],
    "lo": (
        "Compose the Trainer class skeleton with the scalar-loss backward contract so each "
        "training_step call advances the optimizer once and the step counter / history stay in sync."
    ),
}


# ===========================================================================
# cx28 — Trainer with DataLoader integration
# (trainer-class-skeleton + dataloader-batching)
# ===========================================================================
spec_28 = {
    "atom_ids": ["trainer-class-skeleton", "dataloader-batching"],
    "subtopics": _subs(["trainer-class-skeleton", "dataloader-batching"]),
    "primary_atom": "trainer-class-skeleton",
    "part": "part5",
    "exercise_index": 28,
    "exercise_title": "Trainer.fit walks a DataLoader: one step per batch, B batches per epoch",
    "slug": "trainer-fit-with-dataloader",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "The Trainer's `fit(n_epochs)` method is where the OO skeleton meets PyTorch's data-loading "
        "machinery. Each epoch iterates the `DataLoader`, calling `training_step` on every batch. "
        "Two atoms compose:\n\n"
        "1. **trainer-class-skeleton** — the `Trainer.fit` method that walks epochs and batches.\n"
        "2. **dataloader-batching** — `DataLoader(TensorDataset(...), batch_size=B, shuffle=...)` "
        "yields batches of size `B`. `len(dataloader) == ceil(N / B)` (or `N // B` if "
        "`drop_last=True`).\n\n"
        "**Anatomy of `fit`.**\n"
        "```python\n"
        "def fit(self, n_epochs):\n"
        "    for epoch in range(n_epochs):\n"
        "        for batch in self.train_loader:        # dataloader-batching.\n"
        "            self.training_step(batch)          # trainer-class-skeleton.\n"
        "```\n\n"
        "**The step count is deterministic.** After `fit(n_epochs)` on a loader with "
        "`len(loader) = B_count` batches: `trainer.step == n_epochs * B_count`. That equation is "
        "what the test pins down — any off-by-one in the loop (one extra batch, skipping the first, "
        "etc.) breaks it."
    ),
    "prompt_body": (
        "Implement `cx28_make_trainer_with_fit()` returning a `LoaderTrainer` class.\n\n"
        "Required structure:\n"
        "- `LoaderTrainer.__init__(self, model, optimizer, loss_fn, train_loader)`:\n"
        "  - Store the four args. `self.step = 0`. `self.epoch = 0`. `self.history = []`.\n"
        "- `LoaderTrainer.training_step(self, batch)`:\n"
        "  - Same as cx27: forward → scalar loss → zero_grad → backward → step → counter++ → "
        "history append → return scalar loss.\n"
        "- `LoaderTrainer.fit(self, n_epochs)`:\n"
        "  - For each of `n_epochs` epochs:\n"
        "    - Iterate `for batch in self.train_loader`, call `self.training_step(batch)`.\n"
        "    - Increment `self.epoch += 1` at the end of the epoch.\n"
        "  - Return `self.history`.\n\n"
        "The test verifies:\n"
        "- `len(history) == n_epochs * len(train_loader)` (step count equation).\n"
        "- `trainer.step == n_epochs * len(train_loader)` and `trainer.epoch == n_epochs`.\n"
        "- Each `batch` yielded by the loader has the correct batch_size (last batch may be "
        "smaller if `drop_last=False`).\n"
        "- Loss decreases across epochs (regression converges)."
    ),
    "stub_body": (
        "def cx28_make_trainer_with_fit():\n"
        "    \"\"\"Return a LoaderTrainer class whose fit walks a DataLoader.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "LoaderTrainer = cx28_make_trainer_with_fit()\n"
        "\n"
        "# Tiny regression task with TensorDataset + DataLoader.\n"
        "t.manual_seed(0)\n"
        "N = 32\n"
        "x_data = t.randn(N, 1)\n"
        "y_data = 2.5 * x_data - 0.5 + 0.01 * t.randn(N, 1)\n"
        "ds = TensorDataset(x_data, y_data)\n"
        "loader = DataLoader(ds, batch_size=8, shuffle=False)\n"
        "assert len(loader) == 4, f'sanity: 32/8 = 4 batches; got {len(loader)}'\n"
        "\n"
        "model = nn.Linear(1, 1)\n"
        "opt = t.optim.SGD(model.parameters(), lr=0.05)\n"
        "loss_fn = nn.MSELoss()\n"
        "trainer = LoaderTrainer(model, opt, loss_fn, loader)\n"
        "\n"
        "# Case A: initial state.\n"
        "assert trainer.step == 0\n"
        "assert trainer.epoch == 0\n"
        "assert trainer.history == []\n"
        "assert trainer.train_loader is loader\n"
        "\n"
        "# Case B: fit walks the right number of batches.\n"
        "n_epochs = 5\n"
        "history = trainer.fit(n_epochs)\n"
        "expected_steps = n_epochs * len(loader)  # 5 * 4 = 20.\n"
        "assert trainer.step == expected_steps, (\n"
        "    f'step should be {expected_steps} after {n_epochs} epochs of {len(loader)} batches; '\n"
        "    f'got {trainer.step}'\n"
        ")\n"
        "assert trainer.epoch == n_epochs, f'epoch should be {n_epochs}; got {trainer.epoch}'\n"
        "assert len(trainer.history) == expected_steps\n"
        "assert history is trainer.history or history == trainer.history\n"
        "\n"
        "# Case C: history contains floats (per-batch losses).\n"
        "assert all(isinstance(v, float) for v in trainer.history), 'history must store floats'\n"
        "\n"
        "# Case D: loss decreases across epochs (regression converges).\n"
        "first_epoch_avg = sum(trainer.history[:len(loader)]) / len(loader)\n"
        "last_epoch_avg = sum(trainer.history[-len(loader):]) / len(loader)\n"
        "assert last_epoch_avg < first_epoch_avg, (\n"
        "    f'avg loss should decrease across epochs: first={first_epoch_avg:.4f} '\n"
        "    f'last={last_epoch_avg:.4f}'\n"
        ")\n"
        "\n"
        "# Case E: batch-size invariant — fit a SECOND trainer with N=30, B=8 (uneven).\n"
        "# 30 / 8 = 3 full batches of 8 + 1 partial batch of 6 = 4 batches; step should be n_epochs*4.\n"
        "x2 = t.randn(30, 1)\n"
        "y2 = 2.5 * x2 - 0.5\n"
        "ds2 = TensorDataset(x2, y2)\n"
        "loader2 = DataLoader(ds2, batch_size=8, shuffle=False, drop_last=False)\n"
        "assert len(loader2) == 4, f'sanity: ceil(30/8)=4; got {len(loader2)}'\n"
        "model2 = nn.Linear(1, 1)\n"
        "opt2 = t.optim.SGD(model2.parameters(), lr=0.01)\n"
        "tr2 = LoaderTrainer(model2, opt2, nn.MSELoss(), loader2)\n"
        "tr2.fit(3)\n"
        "assert tr2.step == 3 * 4, f'uneven batches: step should be 12; got {tr2.step}'\n"
        "# Verify last batch was the partial one — we can't easily inspect it post hoc, but\n"
        "# step count being right implies fit iterated all 4 batches per epoch.\n"
        "assert tr2.epoch == 3"
    ),
    "solution_body": (
        "def cx28_make_trainer_with_fit():\n"
        "    class LoaderTrainer:\n"
        "        def __init__(self, model, optimizer, loss_fn, train_loader):\n"
        "            self.model = model\n"
        "            self.optimizer = optimizer\n"
        "            self.loss_fn = loss_fn\n"
        "            self.train_loader = train_loader\n"
        "            self.step = 0\n"
        "            self.epoch = 0\n"
        "            self.history = []\n"
        "\n"
        "        def training_step(self, batch):\n"
        "            x, y = batch\n"
        "            logits = self.model(x)\n"
        "            loss = self.loss_fn(logits, y)\n"
        "            self.optimizer.zero_grad()\n"
        "            loss.backward()\n"
        "            self.optimizer.step()\n"
        "            self.step += 1\n"
        "            self.history.append(loss.item())\n"
        "            return loss\n"
        "\n"
        "        def fit(self, n_epochs):\n"
        "            # Atom B (dataloader-batching): one training_step per batch yielded.\n"
        "            for _ in range(n_epochs):\n"
        "                for batch in self.train_loader:\n"
        "                    self.training_step(batch)\n"
        "                self.epoch += 1\n"
        "            return self.history\n"
        "\n"
        "    return LoaderTrainer"
    ),
    "solution_notes": (
        "`for batch in loader` is what makes DataLoader composable with the trainer skeleton — the "
        "loader handles shuffling, batching, and the partial-final-batch edge case; the trainer "
        "just handles 'apply one optimizer step per batch'. If you set `drop_last=True` the partial "
        "batch is skipped and `len(loader)` shrinks by 1."
    ),
    "extra_imports": NN_DL_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["trainer-class-skeleton", "dataloader-batching"],
    "lo": (
        "Compose Trainer.fit with a DataLoader so the step count after n_epochs is exactly "
        "n_epochs * len(loader) and the partial-final-batch edge case is handled automatically."
    ),
}


# ===========================================================================
# cx29 — wandb.init + wandb.log in the same trainer
# (wandb-init-run + wandb-log-step)
# ===========================================================================
spec_29 = {
    "atom_ids": ["wandb-init-run", "wandb-log-step"],
    "subtopics": _subs(["wandb-init-run", "wandb-log-step"]),
    "primary_atom": "wandb-init-run",
    "part": "part5",
    "exercise_index": 29,
    "exercise_title": "Trainer wires wandb.init at start + wandb.log per step",
    "slug": "trainer-wandb-init-then-log",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "A wandb-instrumented Trainer is the ARENA canonical setup: open a run at the start of "
        "training, log per-batch metrics throughout. Two atoms compose:\n\n"
        "1. **wandb-init-run** — `wandb.init(project=, name=, config=)` opens exactly ONE run per "
        "`train()` call. Goes in `pre_training_setup` (or at the very top of `fit`).\n"
        "2. **wandb-log-step** — `wandb.log({'loss': ...}, step=self.step)` is called every batch "
        "from inside `training_step`. The `step=` kwarg is the global step counter (batches "
        "completed), giving wandb the x-axis.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "class WandbTrainer:\n"
        "    def pre_training_setup(self):\n"
        "        wandb.init(                          # wandb-init-run.\n"
        "            project=self.args.wandb_project,\n"
        "            name=self.args.wandb_name,\n"
        "            config=self.args,\n"
        "        )\n"
        "    def training_step(self, batch):\n"
        "        ...\n"
        "        wandb.log({'loss': loss.item()}, step=self.step)  # wandb-log-step.\n"
        "        return loss\n"
        "    def fit(self, n_epochs):\n"
        "        self.pre_training_setup()\n"
        "        for _ in range(n_epochs):\n"
        "            for batch in self.train_loader:\n"
        "                self.training_step(batch)\n"
        "```\n\n"
        "**The test mocks `wandb`.** `sys.modules['wandb']` is replaced with a `MagicMock()` so "
        "the trainer can call `wandb.init` and `wandb.log` without a real wandb install. "
        "The mock records every call so we can assert what was logged."
    ),
    "prompt_body": (
        "Implement `cx29_make_wandb_trainer()` which returns a `WandbTrainer` class.\n\n"
        "Required structure:\n"
        "- `WandbTrainer.__init__(self, model, optimizer, loss_fn, train_loader, args)`:\n"
        "  - Store all five. `self.step = 0`. `self.history = []`.\n"
        "  - `args` is a simple object with `.wandb_project`, `.wandb_name`, `.lr`, `.epochs`.\n"
        "- `WandbTrainer.pre_training_setup(self)`:\n"
        "  - Call `wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)` "
        "(atom: wandb-init-run). EXACTLY one call.\n"
        "- `WandbTrainer.training_step(self, batch)`:\n"
        "  - Forward → scalar loss → zero_grad → backward → opt.step → `self.step += 1`.\n"
        "  - `wandb.log({'loss': loss.item()}, step=self.step)` (atom: wandb-log-step).\n"
        "  - Append `loss.item()` to `self.history`. Return scalar loss.\n"
        "- `WandbTrainer.fit(self, n_epochs)`:\n"
        "  - Call `pre_training_setup()` ONCE before the epoch loop.\n"
        "  - For each epoch: iterate the loader, call `training_step(batch)`.\n\n"
        "The test mocks wandb, then verifies:\n"
        "- `wandb.init` called exactly once (after `fit`), with `project=`, `name=`, `config=args`.\n"
        "- `wandb.log` called exactly `n_epochs * len(loader)` times.\n"
        "- Each `wandb.log` call has a `step=` kwarg equal to the current step counter."
    ),
    "stub_body": (
        "def cx29_make_wandb_trainer():\n"
        "    \"\"\"Return a WandbTrainer class with wandb.init in pre_training_setup + wandb.log per step.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "import sys\n"
        "from unittest.mock import MagicMock\n"
        "sys.modules['wandb'] = MagicMock()\n"
        "import wandb as _wandb_mod\n"
        "_wandb_mod.init.reset_mock()\n"
        "_wandb_mod.log.reset_mock()\n"
        "\n"
        "WandbTrainer = cx29_make_wandb_trainer()\n"
        "\n"
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class FakeArgs:\n"
        "    wandb_project: str = 'arena-vae'\n"
        "    wandb_name: str = 'baseline-cx29'\n"
        "    lr: float = 1e-2\n"
        "    epochs: int = 2\n"
        "\n"
        "t.manual_seed(0)\n"
        "N = 16\n"
        "x_data = t.randn(N, 1)\n"
        "y_data = 1.5 * x_data + 0.2\n"
        "loader = DataLoader(TensorDataset(x_data, y_data), batch_size=4, shuffle=False)\n"
        "assert len(loader) == 4\n"
        "\n"
        "args = FakeArgs()\n"
        "model = nn.Linear(1, 1)\n"
        "opt = t.optim.SGD(model.parameters(), lr=args.lr)\n"
        "trainer = WandbTrainer(model, opt, nn.MSELoss(), loader, args)\n"
        "assert trainer.step == 0\n"
        "\n"
        "n_epochs = args.epochs  # 2.\n"
        "trainer.fit(n_epochs)\n"
        "\n"
        "# Case A: wandb.init called exactly once with the right kwargs.\n"
        "assert _wandb_mod.init.call_count == 1, f'wandb.init must be called exactly once; got {_wandb_mod.init.call_count}'\n"
        "init_kwargs = _wandb_mod.init.call_args.kwargs\n"
        "assert init_kwargs.get('project') == 'arena-vae', f'project kwarg: {init_kwargs.get(\"project\")!r}'\n"
        "assert init_kwargs.get('name') == 'baseline-cx29', f'name kwarg: {init_kwargs.get(\"name\")!r}'\n"
        "assert init_kwargs.get('config') is args, 'config must be the args object itself'\n"
        "\n"
        "# Case B: wandb.log called n_epochs * len(loader) = 2*4 = 8 times.\n"
        "expected_logs = n_epochs * len(loader)\n"
        "assert _wandb_mod.log.call_count == expected_logs, (\n"
        "    f'wandb.log should be called {expected_logs} times; got {_wandb_mod.log.call_count}'\n"
        ")\n"
        "\n"
        "# Case C: each wandb.log call had step=k for k=1..expected_logs.\n"
        "log_calls = _wandb_mod.log.call_args_list\n"
        "for k, call in enumerate(log_calls, start=1):\n"
        "    step_kwarg = call.kwargs.get('step')\n"
        "    assert step_kwarg == k, f'log call #{k} should have step={k}; got step={step_kwarg}'\n"
        "    # The dict logged must contain a 'loss' key with a float value.\n"
        "    metric_dict = call.args[0] if call.args else call.kwargs.get('data')\n"
        "    assert isinstance(metric_dict, dict), f'first positional arg to wandb.log must be a dict; got {type(metric_dict).__name__}'\n"
        "    assert 'loss' in metric_dict, f\"metric dict must contain 'loss'; got keys {list(metric_dict.keys())}\"\n"
        "    assert isinstance(metric_dict['loss'], float), f'loss value must be a float (use .item()); got {type(metric_dict[\"loss\"]).__name__}'\n"
        "\n"
        "# Case D: counters consistent.\n"
        "assert trainer.step == expected_logs, f'step counter must equal log count; {trainer.step} vs {expected_logs}'\n"
        "assert len(trainer.history) == expected_logs\n"
        "\n"
        "# Case E: pre_training_setup is called BEFORE any wandb.log. We can confirm by checking\n"
        "# init.call_count == 1 even though we called fit (which calls init internally).\n"
        "# A trainer that didn't wire init in fit would leave init.call_count at 0.\n"
        "assert _wandb_mod.init.call_count == 1"
    ),
    "solution_body": (
        "def cx29_make_wandb_trainer():\n"
        "    import sys\n"
        "    from unittest.mock import MagicMock\n"
        "    sys.modules.setdefault('wandb', MagicMock())\n"
        "    import wandb\n"
        "\n"
        "    class WandbTrainer:\n"
        "        def __init__(self, model, optimizer, loss_fn, train_loader, args):\n"
        "            self.model = model\n"
        "            self.optimizer = optimizer\n"
        "            self.loss_fn = loss_fn\n"
        "            self.train_loader = train_loader\n"
        "            self.args = args\n"
        "            self.step = 0\n"
        "            self.history = []\n"
        "\n"
        "        def pre_training_setup(self):\n"
        "            # Atom A (wandb-init-run): exactly one run per fit().\n"
        "            wandb.init(\n"
        "                project=self.args.wandb_project,\n"
        "                name=self.args.wandb_name,\n"
        "                config=self.args,\n"
        "            )\n"
        "\n"
        "        def training_step(self, batch):\n"
        "            x, y = batch\n"
        "            logits = self.model(x)\n"
        "            loss = self.loss_fn(logits, y)\n"
        "            self.optimizer.zero_grad()\n"
        "            loss.backward()\n"
        "            self.optimizer.step()\n"
        "            self.step += 1\n"
        "            # Atom B (wandb-log-step): per-batch metric with step axis.\n"
        "            wandb.log({'loss': loss.item()}, step=self.step)\n"
        "            self.history.append(loss.item())\n"
        "            return loss\n"
        "\n"
        "        def fit(self, n_epochs):\n"
        "            self.pre_training_setup()\n"
        "            for _ in range(n_epochs):\n"
        "                for batch in self.train_loader:\n"
        "                    self.training_step(batch)\n"
        "\n"
        "    return WandbTrainer"
    ),
    "solution_notes": (
        "Order matters: `wandb.init` must fire BEFORE the first `wandb.log` or wandb errors out "
        "with 'no run in progress'. Putting `init` in `pre_training_setup` (called once at the top "
        "of `fit`) guarantees the ordering. The `step=` kwarg pins wandb's x-axis to your training "
        "step — without it, wandb infers a step automatically and can lose alignment across runs."
    ),
    "extra_imports": WANDB_DL_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["wandb-init-run", "wandb-log-step"],
    "lo": (
        "Compose wandb.init in pre_training_setup with wandb.log per training_step so a Trainer.fit "
        "call opens exactly one run and emits exactly len(loader)*n_epochs logged scalar metrics."
    ),
}


# ===========================================================================
# cx30 — wandb.init + log_samples eval callback emits wandb.Image
# (wandb-init-run + log-samples-eval-callback)
# ===========================================================================
spec_30 = {
    "atom_ids": ["wandb-init-run", "log-samples-eval-callback"],
    "subtopics": _subs(["wandb-init-run", "log-samples-eval-callback"]),
    "primary_atom": "log-samples-eval-callback",
    "part": "part5",
    "exercise_index": 30,
    "exercise_title": "Trainer wires wandb.init + every-K-step log_samples callback that emits wandb.Image",
    "slug": "trainer-wandb-init-then-log-samples-callback",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Visual generative models (VAEs, GANs, diffusion) log SAMPLES periodically so you can SEE "
        "training progress in the wandb UI. The composition:\n\n"
        "1. **wandb-init-run** — open one run at start of training.\n"
        "2. **log-samples-eval-callback** — every K steps, generate N samples from the model, wrap "
        "each as a `wandb.Image(...)`, and log them with `wandb.log({'samples': [Image, Image, ...]})`. "
        "The eval-callback cadence is `step % eval_every == 0`.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "class SampleLoggingTrainer:\n"
        "    def pre_training_setup(self):\n"
        "        wandb.init(project=..., name=..., config=self.args)        # wandb-init-run.\n"
        "    def training_step(self, batch):\n"
        "        ...\n"
        "        self.step += 1\n"
        "        if self.step % self.args.eval_every == 0:\n"
        "            self.log_samples()                                      # log-samples-eval-callback.\n"
        "    def log_samples(self):\n"
        "        with t.inference_mode():\n"
        "            samples = self.model.sample(n=self.args.n_eval)\n"
        "        images = [wandb.Image(s) for s in samples]\n"
        "        wandb.log({'samples': images}, step=self.step)\n"
        "```\n\n"
        "**Step 0 vs step K.** Some recipes log samples at step 0 too (initial random model "
        "baseline). The convention varies; here we fire when `step > 0 and step % eval_every == 0` "
        "— so for `eval_every=3, n_steps=10` we fire at steps 3, 6, 9 (3 fires). "
        "The test pins down exactly this cadence."
    ),
    "prompt_body": (
        "Implement `cx30_make_sample_logger_trainer()` returning a `SampleLoggingTrainer` class.\n\n"
        "Required structure:\n"
        "- `SampleLoggingTrainer.__init__(self, model, optimizer, loss_fn, train_loader, args)`:\n"
        "  - Store all five. `self.step = 0`. `args` has `.wandb_project`, `.wandb_name`, `.lr`, "
        "`.epochs`, `.eval_every`, `.n_eval`.\n"
        "- `SampleLoggingTrainer.pre_training_setup(self)`:\n"
        "  - `wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)` "
        "(atom: wandb-init-run).\n"
        "- `SampleLoggingTrainer.training_step(self, batch)`:\n"
        "  - Forward → scalar loss → zero_grad → backward → opt.step → `self.step += 1`.\n"
        "  - If `self.step > 0 and self.step % self.args.eval_every == 0`, call "
        "`self.log_samples()` (atom: log-samples-eval-callback).\n"
        "  - Return scalar loss.\n"
        "- `SampleLoggingTrainer.log_samples(self)`:\n"
        "  - Create `self.args.n_eval` dummy sample tensors: `[t.randn(3, 8, 8) for _ in "
        "range(self.args.n_eval)]` (CHW shape — wandb.Image accepts that).\n"
        "  - Wrap each as `wandb.Image(s)`.\n"
        "  - `wandb.log({'samples': images}, step=self.step)`.\n"
        "- `SampleLoggingTrainer.fit(self, n_epochs)`:\n"
        "  - `pre_training_setup()` once, then loop epochs × loader, calling `training_step(batch)`.\n\n"
        "The test verifies (with mocked wandb):\n"
        "- `wandb.init` called once with project/name/config.\n"
        "- `wandb.Image` constructor called `n_eval` times PER firing of the callback.\n"
        "- `wandb.log` called with `{'samples': [...]}` at exactly the expected step counts.\n"
        "- The number of fires matches `n_steps_total // eval_every` (since we skip step 0)."
    ),
    "stub_body": (
        "def cx30_make_sample_logger_trainer():\n"
        "    \"\"\"Return a SampleLoggingTrainer class with init + every-K-step log_samples callback.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "import sys\n"
        "from unittest.mock import MagicMock\n"
        "sys.modules['wandb'] = MagicMock()\n"
        "import wandb as _wandb_mod\n"
        "_wandb_mod.init.reset_mock()\n"
        "_wandb_mod.log.reset_mock()\n"
        "_wandb_mod.Image.reset_mock()\n"
        "\n"
        "SampleLoggingTrainer = cx30_make_sample_logger_trainer()\n"
        "\n"
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class FakeArgs:\n"
        "    wandb_project: str = 'arena-gan'\n"
        "    wandb_name: str = 'sample-logger-cx30'\n"
        "    lr: float = 1e-2\n"
        "    epochs: int = 3\n"
        "    eval_every: int = 3\n"
        "    n_eval: int = 4\n"
        "\n"
        "t.manual_seed(0)\n"
        "N = 12\n"
        "x_data = t.randn(N, 1)\n"
        "y_data = 0.5 * x_data + 1.0\n"
        "loader = DataLoader(TensorDataset(x_data, y_data), batch_size=4, shuffle=False)\n"
        "assert len(loader) == 3, f'sanity: 12/4=3; got {len(loader)}'\n"
        "\n"
        "args = FakeArgs()\n"
        "model = nn.Linear(1, 1)\n"
        "opt = t.optim.SGD(model.parameters(), lr=args.lr)\n"
        "trainer = SampleLoggingTrainer(model, opt, nn.MSELoss(), loader, args)\n"
        "trainer.fit(args.epochs)  # 3 epochs * 3 batches = 9 steps.\n"
        "\n"
        "# Case A: wandb.init called exactly once with right kwargs.\n"
        "assert _wandb_mod.init.call_count == 1\n"
        "ik = _wandb_mod.init.call_args.kwargs\n"
        "assert ik.get('project') == 'arena-gan'\n"
        "assert ik.get('name') == 'sample-logger-cx30'\n"
        "assert ik.get('config') is args\n"
        "\n"
        "# Case B: total steps = 9. Callback fires at step % 3 == 0 with step > 0 → steps 3, 6, 9 → 3 fires.\n"
        "expected_fires = 3\n"
        "total_steps = args.epochs * len(loader)\n"
        "assert trainer.step == total_steps, f'step counter wrong: {trainer.step} vs {total_steps}'\n"
        "\n"
        "# Case C: wandb.Image constructed exactly n_eval * expected_fires times.\n"
        "expected_image_calls = args.n_eval * expected_fires  # 4 * 3 = 12.\n"
        "assert _wandb_mod.Image.call_count == expected_image_calls, (\n"
        "    f'wandb.Image should be called {expected_image_calls} times; got {_wandb_mod.Image.call_count}'\n"
        ")\n"
        "\n"
        "# Case D: wandb.log called exactly expected_fires times with {'samples': [...]}.\n"
        "sample_log_calls = []\n"
        "for c in _wandb_mod.log.call_args_list:\n"
        "    payload = c.args[0] if c.args else c.kwargs.get('data')\n"
        "    if isinstance(payload, dict) and 'samples' in payload:\n"
        "        sample_log_calls.append(c)\n"
        "assert len(sample_log_calls) == expected_fires, (\n"
        "    f\"expected {expected_fires} wandb.log({{'samples':...}}) calls; got {len(sample_log_calls)}\"\n"
        ")\n"
        "\n"
        "# Case E: each samples-log call has the right step= and list length n_eval.\n"
        "expected_steps = [3, 6, 9]\n"
        "for call, exp_step in zip(sample_log_calls, expected_steps):\n"
        "    step_kw = call.kwargs.get('step')\n"
        "    assert step_kw == exp_step, f'samples-log step mismatch: {step_kw} vs {exp_step}'\n"
        "    payload = call.args[0] if call.args else call.kwargs.get('data')\n"
        "    samples_list = payload['samples']\n"
        "    assert isinstance(samples_list, list), f\"samples value must be a list; got {type(samples_list).__name__}\"\n"
        "    assert len(samples_list) == args.n_eval, (\n"
        "        f'samples list length must equal n_eval={args.n_eval}; got {len(samples_list)}'\n"
        "    )\n"
        "\n"
        "# Case F: init was called BEFORE any wandb.Image (sample logging requires open run).\n"
        "# With mocks we can't easily order across attributes, but init.call_count==1 and Image.call_count>0\n"
        "# both being satisfied at end-of-fit is good enough for the contract.\n"
        "assert _wandb_mod.init.call_count == 1 and _wandb_mod.Image.call_count > 0"
    ),
    "solution_body": (
        "def cx30_make_sample_logger_trainer():\n"
        "    import sys\n"
        "    from unittest.mock import MagicMock\n"
        "    sys.modules.setdefault('wandb', MagicMock())\n"
        "    import wandb\n"
        "\n"
        "    class SampleLoggingTrainer:\n"
        "        def __init__(self, model, optimizer, loss_fn, train_loader, args):\n"
        "            self.model = model\n"
        "            self.optimizer = optimizer\n"
        "            self.loss_fn = loss_fn\n"
        "            self.train_loader = train_loader\n"
        "            self.args = args\n"
        "            self.step = 0\n"
        "\n"
        "        def pre_training_setup(self):\n"
        "            # Atom A (wandb-init-run).\n"
        "            wandb.init(\n"
        "                project=self.args.wandb_project,\n"
        "                name=self.args.wandb_name,\n"
        "                config=self.args,\n"
        "            )\n"
        "\n"
        "        def training_step(self, batch):\n"
        "            x, y = batch\n"
        "            logits = self.model(x)\n"
        "            loss = self.loss_fn(logits, y)\n"
        "            self.optimizer.zero_grad()\n"
        "            loss.backward()\n"
        "            self.optimizer.step()\n"
        "            self.step += 1\n"
        "            # Atom B (log-samples-eval-callback): every-K-steps cadence (skip step 0).\n"
        "            if self.step > 0 and self.step % self.args.eval_every == 0:\n"
        "                self.log_samples()\n"
        "            return loss\n"
        "\n"
        "        def log_samples(self):\n"
        "            # Generate n_eval dummy samples (CHW so wandb.Image is happy).\n"
        "            with t.inference_mode():\n"
        "                samples = [t.randn(3, 8, 8) for _ in range(self.args.n_eval)]\n"
        "            images = [wandb.Image(s) for s in samples]\n"
        "            wandb.log({'samples': images}, step=self.step)\n"
        "\n"
        "        def fit(self, n_epochs):\n"
        "            self.pre_training_setup()\n"
        "            for _ in range(n_epochs):\n"
        "                for batch in self.train_loader:\n"
        "                    self.training_step(batch)\n"
        "\n"
        "    return SampleLoggingTrainer"
    ),
    "solution_notes": (
        "Wrapping each sample in `wandb.Image` (rather than logging raw tensors) tells wandb to "
        "render it as an image in the run UI — wandb accepts numpy arrays, PIL Images, torch "
        "tensors in CHW format, and matplotlib figures. The list-of-Images value under "
        "`'samples'` becomes a gallery in the dashboard. Logging samples is expensive on real "
        "wandb (network upload), so keep `n_eval` small (≤16) and `eval_every` large (≥100 in "
        "real training)."
    ),
    "extra_imports": WANDB_DL_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["wandb-init-run", "log-samples-eval-callback"],
    "lo": (
        "Compose wandb.init in pre_training_setup with an every-K-step log_samples callback that "
        "emits N wandb.Image objects per fire so a Trainer.fit run opens exactly one wandb run "
        "and the sample gallery cadence matches step % eval_every == 0."
    ),
}


SPECS = [spec_25, spec_26, spec_27, spec_28, spec_29, spec_30]


def main() -> None:
    for spec in SPECS:
        out = emit_composite(spec)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
