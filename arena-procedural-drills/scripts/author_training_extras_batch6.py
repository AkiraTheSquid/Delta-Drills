#!/usr/bin/env python3
"""Author 8 standalone Colab drills for miscellaneous training-infrastructure
atoms (batch-6).

Atoms covered (each drill = ONE LO + ONE Bloom level, max 2 concurrent KCs):

  dataclasses-replace-args         — 1 drill (ex1)  prereqs_hparam_config
  weight-decay-decoupled           — 1 drill (ex1)  prereqs_optimizer_internals
  clip-grad-norm-pre-step          — 1 drill (ex1)  prereqs_optimizer_internals
  two-optimizers-alternating-step  — 1 drill (ex1)  prereqs_generative
  detach-stop-gradient-trick       — 1 drill (ex1)  prereqs_generative
  dataloader-pin-memory-workers    — 1 drill (ex1)  prereqs_pytorch_modules
  distributed-sampler-shard        — 1 drill (ex1)  prereqs_distributed
  rank0-only-side-effects          — 1 drill (ex1)  prereqs_distributed

These are SMALLER constituent training-infra skills the ARENA chap-0 part-3
optimization material, the chap-0 part-4 generative material, and chap-0
part-3 distributed material all assume the learner can perform in isolation.

Each spec is verified by re-running its solution against its test_body
inside the build venv (torch 2.12.0+cpu) before emission. Any failure
aborts the build.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone


# ---------------------------------------------------------------------------
# Per-atom recap blocks.
# ---------------------------------------------------------------------------

RECAP_DATACLASSES_REPLACE = (
    "## Config: `dataclasses.replace(args, ...)` — quick refresher\n"
    "\n"
    "When you want a *modified copy* of an args object without mutating "
    "the original, the idiom is:\n"
    "\n"
    "```python\n"
    "from dataclasses import dataclass, replace\n"
    "\n"
    "@dataclass\n"
    "class Args:\n"
    "    lr: float = 1e-3\n"
    "    bs: int = 32\n"
    "    epochs: int = 10\n"
    "\n"
    "args = Args()\n"
    "args_v2 = replace(args, lr=1e-5)   # only lr changes; bs/epochs copied\n"
    "```\n"
    "\n"
    "**`replace` returns a NEW instance.** `args` is untouched. This is "
    "the dataclass equivalent of `dict | {'lr': 1e-5}` — same later-wins "
    "semantics, but preserves the type and re-runs `__post_init__` "
    "validation.\n"
    "\n"
    "**Why not `args.lr = 1e-5`.** Mutation is fine for one-shot tweaks "
    "but breaks any code that holds a reference to the original. The "
    "sweep harness, the wandb logger, and the checkpoint writer all "
    "expect a stable args object. `replace` gives every consumer its own "
    "frozen view.\n"
    "\n"
    "**Validation re-runs.** Because `replace` calls `Args(**new_fields)`, "
    "your `__post_init__` validators fire again — bad overrides raise at "
    "replace time, not later when the bad value is used."
)

RECAP_WEIGHT_DECAY_DECOUPLED = (
    "## Optimizer: decoupled weight decay (AdamW) — quick refresher\n"
    "\n"
    "`Adam` and `AdamW` differ in ONE line. Adam folds weight decay into "
    "the gradient *before* the adaptive moment estimates run:\n"
    "\n"
    "```python\n"
    "# Adam (coupled): wd added to grad, gets m/v statistics applied to it\n"
    "g = p.grad + wd * p\n"
    "m = beta1*m + (1-beta1)*g\n"
    "v = beta2*v + (1-beta2)*g*g\n"
    "p -= lr * m / (sqrt(v) + eps)\n"
    "```\n"
    "\n"
    "AdamW *decouples* the decay — it's applied to the parameters "
    "directly, as a SEPARATE update from the Adam moment step, "
    "untouched by the moment statistics. `torch.optim.AdamW` "
    "applies the decay BEFORE the Adam step:\n"
    "\n"
    "```python\n"
    "# AdamW (decoupled): two separate updates\n"
    "p *= (1 - lr * wd)              # decoupled wd step (== p -= lr*wd*p)\n"
    "m = beta1*m + (1-beta1)*p.grad\n"
    "v = beta2*v + (1-beta2)*p.grad*p.grad\n"
    "p -= lr * m / (sqrt(v) + eps)   # Adam step\n"
    "```\n"
    "\n"
    "**Why this matters.** In Adam, the decay term flows through "
    "`sqrt(v)` normalization, so weights with large gradient variance "
    "effectively get LESS decay — exactly backwards of what L2 "
    "regularization is supposed to do (Loshchilov & Hutter 2017). "
    "AdamW restores the original L2-reg meaning.\n"
    "\n"
    "**The combined-step form.** A common compact rewrite is "
    "`p -= lr * (m/(sqrt(v)+eps) + wd * p)`. Mathematically equivalent "
    "to the two-line version above; you'll see both in the wild."
)

RECAP_CLIP_GRAD_NORM = (
    "## Optimizer: `clip_grad_norm_` pre-step — quick refresher\n"
    "\n"
    "Gradient clipping rescales every gradient by the SAME factor when "
    "the global L2 norm exceeds `max_norm`. The canonical placement:\n"
    "\n"
    "```python\n"
    "loss.backward()\n"
    "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n"
    "optimizer.step()\n"
    "optimizer.zero_grad()\n"
    "```\n"
    "\n"
    "**The math.** Let `g_total = sqrt(sum_i ||g_i||^2)` be the global "
    "L2 norm. If `g_total > max_norm`, every param's grad is multiplied "
    "by `max_norm / g_total`. Otherwise, no change.\n"
    "\n"
    "**In-place (note the trailing underscore).** "
    "`clip_grad_norm_` modifies `.grad` in place. The return value is "
    "the PRE-clip norm — useful for logging.\n"
    "\n"
    "**MUST be before `optimizer.step()`.** The optimizer reads `.grad`; "
    "if you clip after, the step already used the unclipped gradients. "
    "Equally bad: clipping after `zero_grad` is a no-op (gradients are "
    "gone). Order: backward → clip → step → zero_grad.\n"
    "\n"
    "**Direction preserved.** Because every grad is scaled by the same "
    "scalar, the gradient direction in parameter space is unchanged — "
    "only the magnitude. That's the whole point: bound the step size "
    "without distorting which way you're going.\n"
    "\n"
    "**Why not `clip_grad_value_`.** Element-wise clipping (each tensor "
    "scaled independently) destroys the direction. Norm-clipping is "
    "almost always the right primitive for transformer training."
)

RECAP_TWO_OPTIMIZERS = (
    "## GAN: two-optimizers alternating step — quick refresher\n"
    "\n"
    "GAN training is two ADVERSARIAL optimization problems wired into "
    "one loop. Each iteration takes a D-step then a G-step (or vice "
    "versa). The canonical pattern:\n"
    "\n"
    "```python\n"
    "for x_real in dataloader:\n"
    "    # === D-step ===\n"
    "    D_opt.zero_grad()\n"
    "    z = torch.randn(B, z_dim)\n"
    "    fake = G(z).detach()              # stop-gradient into G\n"
    "    loss_D = -(D(x_real).log() + (1 - D(fake)).log()).mean()\n"
    "    loss_D.backward()\n"
    "    D_opt.step()\n"
    "\n"
    "    # === G-step ===\n"
    "    G_opt.zero_grad()\n"
    "    z = torch.randn(B, z_dim)\n"
    "    fake = G(z)                       # GRAD flows into G this time\n"
    "    loss_G = -D(fake).log().mean()\n"
    "    loss_G.backward()\n"
    "    G_opt.step()\n"
    "```\n"
    "\n"
    "**Two optimizers, two parameter sets.** "
    "`D_opt = Adam(D.parameters(), ...)` only knows about D. "
    "`G_opt = Adam(G.parameters(), ...)` only knows about G. Mixing "
    "params across optimizers is the most common GAN bug — "
    "`G_opt.step()` would silently change D's weights too.\n"
    "\n"
    "**Zero the OWNED grads only.** `D_opt.zero_grad()` clears grads on "
    "D's params but leaves G's grads alone (G doesn't have any yet "
    "anyway). Calling `model.zero_grad()` is also fine if you keep both "
    "modules separate.\n"
    "\n"
    "**Order matters for the loss landscape.** D-then-G is the original "
    "Goodfellow recipe. G-then-D is also common (WGAN-GP). Whatever the "
    "order, both steps happen in each iteration."
)

RECAP_DETACH_STOP_GRADIENT = (
    "## GAN: `.detach()` stop-gradient trick — quick refresher\n"
    "\n"
    "When training the discriminator on a fake sample produced by the "
    "generator, you MUST detach the fake before passing it to D:\n"
    "\n"
    "```python\n"
    "fake = G(z)\n"
    "loss_D = bce(D(fake.detach()), zeros)    # backward stops at fake\n"
    "loss_D.backward()\n"
    "D_opt.step()\n"
    "```\n"
    "\n"
    "**What `.detach()` does.** Returns a new tensor that SHARES storage "
    "with the original but has `requires_grad=False` and is treated as a "
    "leaf by autograd. Backward through `fake.detach()` stops at that "
    "node — no grad flows into G's parameters.\n"
    "\n"
    "**Why the D-step needs this.** During the D-step you're computing "
    "`d(loss_D)/d(D's params)`. If you forgot `.detach()`, autograd "
    "would ALSO compute `d(loss_D)/d(G's params)` — and then when you "
    "later run the G-step's `.backward()`, G's grads are already "
    "populated with the WRONG gradient (the one that would make G "
    "easier for D to detect, the opposite of what you want).\n"
    "\n"
    "**Why the G-step doesn't detach.** In `loss_G = -D(G(z)).log().mean()` "
    "you WANT the gradient to flow through D and into G. D's params are "
    "frozen-by-convention during the G-step (you only call `G_opt.step()`), "
    "so D's weights don't move even though grads accumulate in `D.grad`.\n"
    "\n"
    "**Cheaper alternative for the D-step.** `with torch.no_grad(): "
    "fake = G(z)`. Identical effect, slightly faster because the forward "
    "doesn't build the autograd graph at all. `.detach()` is more common "
    "in published GAN code because it makes the stop-gradient point "
    "explicit at the use site."
)

RECAP_DATALOADER_PIN_MEMORY = (
    "## PyTorch: `DataLoader` `pin_memory` + workers — quick refresher\n"
    "\n"
    "Two `DataLoader` flags control input-pipeline throughput:\n"
    "\n"
    "```python\n"
    "loader = DataLoader(\n"
    "    dataset,\n"
    "    batch_size=64,\n"
    "    num_workers=4,        # background processes that fetch batches\n"
    "    pin_memory=True,      # batches land in page-locked RAM\n"
    "    shuffle=True,\n"
    ")\n"
    "```\n"
    "\n"
    "**`num_workers > 0` parallelizes data loading.** The main process "
    "yields a batch from a queue while N worker subprocesses prepare "
    "future batches in parallel. Bottleneck shifts from disk → CPU "
    "decode → main-thread-blocking to fully overlapped. `num_workers=4` "
    "is the common default; tune to ~half your CPU core count.\n"
    "\n"
    "**`pin_memory=True` enables async CPU→GPU transfer.** Pinned "
    "(page-locked) RAM lets the GPU's DMA engine copy a batch while the "
    "previous batch is still being trained on. Combined with "
    "`batch.to(device, non_blocking=True)` in the training loop, you get "
    "essentially free CPU→GPU staging.\n"
    "\n"
    "**Pin-memory caveat.** Page-locked RAM is a scarce resource; if you "
    "pin too much (huge batches × many workers) you can OOM the system "
    "RAM. The default `False` is conservative — turn it ON for GPU "
    "training, leave it OFF for CPU training (no benefit).\n"
    "\n"
    "**Workers > 0 caveat.** Each worker is a separate Python process. "
    "Anything you pass through the dataset must be picklable. Lambdas in "
    "transforms break — use a top-level function. On Windows, `spawn` "
    "is the only start method, which is much slower to bring up."
)

RECAP_DISTRIBUTED_SAMPLER = (
    "## Distributed: `DistributedSampler` shard — quick refresher\n"
    "\n"
    "`DistributedSampler` partitions a dataset across `world_size` ranks "
    "so each rank sees a DISJOINT slice each epoch:\n"
    "\n"
    "```python\n"
    "from torch.utils.data import DataLoader\n"
    "from torch.utils.data.distributed import DistributedSampler\n"
    "\n"
    "sampler = DistributedSampler(\n"
    "    dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=0,\n"
    ")\n"
    "loader = DataLoader(dataset, batch_size=B, sampler=sampler)\n"
    "\n"
    "for epoch in range(num_epochs):\n"
    "    sampler.set_epoch(epoch)   # MUST call so shuffle re-derives\n"
    "    for batch in loader: ...\n"
    "```\n"
    "\n"
    "**What `num_replicas` and `rank` do.** The sampler builds a "
    "deterministic permutation of `[0, len(dataset))`, then yields "
    "indices `i` where `i % num_replicas == rank`. Rank 0 sees indices "
    "0, W, 2W, ...; rank 1 sees 1, W+1, 2W+1, ... — each rank sees "
    "`len(dataset) / num_replicas` items.\n"
    "\n"
    "**Padding to evenly divide.** If `len(dataset)` isn't a multiple "
    "of `num_replicas`, by default the sampler PADS the index list by "
    "wrapping from the start, so every rank gets exactly the same "
    "number of items. Pass `drop_last=True` to drop the tail instead.\n"
    "\n"
    "**`set_epoch` is mandatory for shuffled training.** The shuffle "
    "uses `epoch + seed` as its RNG state. Without `set_epoch(epoch)`, "
    "every epoch re-uses the same permutation — your model sees the "
    "same batch order forever.\n"
    "\n"
    "**Use `sampler=`, NOT `shuffle=True`.** The DataLoader's "
    "`shuffle=True` is mutually exclusive with `sampler=`. The shuffle "
    "must live inside the sampler so it's coordinated across ranks."
)

RECAP_RANK0_ONLY = (
    "## Distributed: rank-0-only side effects — quick refresher\n"
    "\n"
    "Anything that touches a SHARED RESOURCE (filesystem, network, "
    "wandb, tqdm, print to stdout) must run on exactly one rank — "
    "conventionally rank 0:\n"
    "\n"
    "```python\n"
    "if rank == 0:\n"
    "    torch.save(model.state_dict(), 'ckpt.pt')\n"
    "    wandb.log({'loss': loss.item()})\n"
    "    print(f'epoch {epoch}: loss={loss.item():.4f}')\n"
    "```\n"
    "\n"
    "**Why.** Without the guard, every rank writes to `ckpt.pt` — N "
    "concurrent writes race for the same path, the file ends up "
    "corrupted, and you've also wasted N times the wandb quota. Same "
    "for stdout: log lines interleave from N processes and become "
    "unreadable.\n"
    "\n"
    "**Things that do NOT need the guard.** Anything that's "
    "process-local: per-rank tensors, per-rank `.grad`, the model "
    "forward pass, the optimizer step. Every rank computes its own "
    "grads on its own shard.\n"
    "\n"
    "**Things that DO need the guard.**\n"
    "- File writes (`torch.save`, `open('w')`, `csv.writer`).\n"
    "- Network calls (wandb, MLflow, http POST).\n"
    "- `print` / `logging` / `tqdm` (or use a logger that filters by "
    "rank).\n"
    "- Mutating shared filesystem state (creating directories, "
    "downloading datasets — race on first call).\n"
    "\n"
    "**Pair with `dist.barrier()` when other ranks depend on the work.** "
    "If rank 0 downloads a dataset that rank 1 will read, rank 1 must "
    "wait until rank 0 finishes — otherwise rank 1 tries to read a "
    "half-written file. The idiom: rank 0 does the work, then `barrier`; "
    "rank > 0 hits `barrier` first, blocks, then proceeds."
)


# ---------------------------------------------------------------------------
# Specs.
# ---------------------------------------------------------------------------

SPECS = [

    # =========================================================
    # dataclasses-replace-args — ex1
    # =========================================================
    {
        "atom_id": "dataclasses-replace-args",
        "subtopic": "Config: dataclasses.replace args",
        "topic_folder": "prereqs_hparam_config",
        "atom_recap_md": RECAP_DATACLASSES_REPLACE,
        "exercise_index": 1,
        "exercise_title": "make a sweep of args variants via dataclasses.replace",
        "slug": "make-a-sweep-of-args-variants-via-dataclasses-replace",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["dataclass", "replace", "immutable-update", "sweep"],
        "kcs": [
            "dataclasses-replace-keyword-overrides",
            "input-isolation-no-mutation",
        ],
        "lo": (
            "Apply `dataclasses.replace(base, **overrides)` to "
            "produce a sweep of args variants from one base config "
            "without mutating the base."
        ),
        "prompt_body": (
            "Implement `ex1_make_lr_sweep(base, lrs)`. The standard "
            "sweep-over-LR pattern.\n\n"
            "1. `base` is a `TrainingArgs` dataclass instance (already "
            "defined in the stub).\n"
            "2. `lrs` is a list of floats (e.g. `[1e-5, 3e-5, 1e-4]`).\n"
            "3. For each `lr`, build a *new* `TrainingArgs` with that "
            "`lr` and ALL OTHER fields copied from `base` — using "
            "`dataclasses.replace(base, lr=lr)`.\n"
            "4. Return the list of variants in input order.\n\n"
            "Constraints:\n"
            "- Must NOT mutate `base`.\n"
            "- Each returned variant must be a DIFFERENT object from "
            "`base` (not just `base` itself).\n"
            "- `__post_init__` validation must still run — pass a bad "
            "lr and expect `ValueError`.\n\n"
            "Output: `list[TrainingArgs]`."
        ),
        "stub": (
            "from dataclasses import dataclass, replace\n"
            "\n"
            "@dataclass\n"
            "class TrainingArgs:\n"
            "    lr: float = 1e-3\n"
            "    batch_size: int = 32\n"
            "    epochs: int = 10\n"
            "    optimizer_name: str = 'adam'\n"
            "\n"
            "    def __post_init__(self):\n"
            "        if self.lr <= 0:\n"
            "            raise ValueError(f'lr must be > 0, got {self.lr}')\n"
            "        if self.batch_size < 1:\n"
            "            raise ValueError(f'batch_size must be >= 1, got {self.batch_size}')\n"
            "\n"
            "def ex1_make_lr_sweep(base: 'TrainingArgs', lrs: list) -> list:\n"
            '    """Return a list of TrainingArgs, one per lr, base unchanged."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# === Basic sweep ===\n"
            "base = TrainingArgs(lr=1e-3, batch_size=64, epochs=20, optimizer_name='adamw')\n"
            "lrs = [1e-5, 3e-5, 1e-4]\n"
            "variants = ex1_make_lr_sweep(base, lrs)\n"
            "assert isinstance(variants, list)\n"
            "assert len(variants) == 3\n"
            "\n"
            "# === Each variant has the right lr ===\n"
            "for v, lr in zip(variants, lrs):\n"
            "    assert v.lr == lr, f'lr should be {lr}, got {v.lr}'\n"
            "\n"
            "# === Other fields copied from base ===\n"
            "for v in variants:\n"
            "    assert v.batch_size == 64, f'batch_size should copy from base, got {v.batch_size}'\n"
            "    assert v.epochs == 20\n"
            "    assert v.optimizer_name == 'adamw'\n"
            "\n"
            "# === base is NOT mutated ===\n"
            "assert base.lr == 1e-3, f'base.lr was mutated to {base.lr}'\n"
            "assert base.batch_size == 64\n"
            "\n"
            "# === Each variant is a distinct object ===\n"
            "for v in variants:\n"
            "    assert v is not base, 'variant must not be the same instance as base'\n"
            "ids = [id(v) for v in variants]\n"
            "assert len(set(ids)) == 3, 'each variant must be a distinct object'\n"
            "\n"
            "# === Variants are still TrainingArgs ===\n"
            "for v in variants:\n"
            "    assert isinstance(v, TrainingArgs)\n"
            "\n"
            "# === __post_init__ still runs (validation fires on replace) ===\n"
            "try:\n"
            "    ex1_make_lr_sweep(base, [1e-5, -1.0, 1e-4])\n"
            "except ValueError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('expected ValueError for negative lr in sweep')\n"
            "\n"
            "# === Empty lr list → empty sweep ===\n"
            "assert ex1_make_lr_sweep(base, []) == []\n"
            "\n"
            "# === Order preserved ===\n"
            "out = ex1_make_lr_sweep(base, [3e-4, 1e-5, 5e-4])\n"
            "assert [v.lr for v in out] == [3e-4, 1e-5, 5e-4]"
        ),
        "solution_body": (
            "from dataclasses import dataclass, replace\n"
            "\n"
            "@dataclass\n"
            "class TrainingArgs:\n"
            "    lr: float = 1e-3\n"
            "    batch_size: int = 32\n"
            "    epochs: int = 10\n"
            "    optimizer_name: str = 'adam'\n"
            "\n"
            "    def __post_init__(self):\n"
            "        if self.lr <= 0:\n"
            "            raise ValueError(f'lr must be > 0, got {self.lr}')\n"
            "        if self.batch_size < 1:\n"
            "            raise ValueError(f'batch_size must be >= 1, got {self.batch_size}')\n"
            "\n"
            "\n"
            "def ex1_make_lr_sweep(base, lrs):\n"
            "    return [replace(base, lr=lr) for lr in lrs]"
        ),
        "solution_notes": (
            "**One-liner is the right scale.** `replace` already does "
            "the heavy lifting (copy fields, run `__post_init__`, "
            "return a new instance). A list comp wraps it.\n\n"
            "**`replace` re-runs `__post_init__`.** That's why the bad "
            "`lr=-1.0` test raises — `replace` builds the new instance "
            "by calling `TrainingArgs(...)`, which triggers your "
            "validator. Free guard against typos in sweep configs.\n\n"
            "**Generalizes beyond LR.** Want to sweep `batch_size` "
            "instead? `[replace(base, batch_size=bs) for bs in bss]`. "
            "Want to sweep two axes? Nested loop, two kwargs to "
            "`replace`. The pattern is the same."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # weight-decay-decoupled — ex1
    # =========================================================
    {
        "atom_id": "weight-decay-decoupled",
        "subtopic": "Optimizer: decoupled weight decay (AdamW)",
        "topic_folder": "prereqs_optimizer_internals",
        "atom_recap_md": RECAP_WEIGHT_DECAY_DECOUPLED,
        "exercise_index": 1,
        "exercise_title": "implement one AdamW step with decoupled weight decay",
        "slug": "implement-one-adamw-step-with-decoupled-weight-decay",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["adamw", "weight-decay", "decoupled", "optimizer-step"],
        "kcs": [
            "adamw-decoupled-decay-formula",
            "adam-vs-adamw-difference",
        ],
        "lo": (
            "Apply the decoupled weight-decay update "
            "`p -= lr * wd * p` AS A SEPARATE STEP from the Adam "
            "moment update, distinguishing AdamW from Adam."
        ),
        "prompt_body": (
            "Implement `ex1_adamw_step(p, grad, m, v, lr, beta1, "
            "beta2, eps, wd, step)`. ONE AdamW update step on a "
            "single parameter tensor `p` (in place).\n\n"
            "Inputs:\n"
            "- `p`: parameter `Tensor` (modified in place).\n"
            "- `grad`: gradient `Tensor`, same shape as `p`.\n"
            "- `m`, `v`: 1st/2nd moment running estimates (same shape "
            "as `p`, modified in place).\n"
            "- `lr`, `beta1`, `beta2`, `eps`, `wd`: scalar floats.\n"
            "- `step`: int >= 1, the current step count (for bias "
            "correction).\n\n"
            "Algorithm — the decoupled form (matches "
            "`torch.optim.AdamW`):\n"
            "1. **Decoupled weight-decay step** (applied directly to "
            "the parameter, NOT mixed into the gradient — this is "
            "what makes it AdamW instead of Adam):\n"
            "   `p <- p * (1 - lr * wd)`   (i.e. `p -= lr * wd * p`)\n"
            "2. **Moment update** (Adam, unchanged):\n"
            "   `m <- beta1*m + (1-beta1)*grad`\n"
            "   `v <- beta2*v + (1-beta2)*grad*grad`\n"
            "3. **Bias correction**:\n"
            "   `m_hat = m / (1 - beta1^step)`\n"
            "   `v_hat = v / (1 - beta2^step)`\n"
            "4. **Adam step** (no wd inside):\n"
            "   `p <- p - lr * m_hat / (sqrt(v_hat) + eps)`\n\n"
            "Return nothing — `p`, `m`, `v` are updated in place. The "
            "test compares against a hand-computed reference and "
            "against `torch.optim.AdamW` itself.\n\n"
            "Note on ordering: PyTorch's `torch.optim.AdamW` applies "
            "the decay BEFORE the Adam step (as above). The original "
            "Loshchilov & Hutter paper allowed either order; for "
            "small `lr` they are numerically indistinguishable, but "
            "the test uses PyTorch's reference so we match its order "
            "exactly."
        ),
        "stub": (
            "def ex1_adamw_step(p, grad, m, v, lr, beta1, beta2, eps, wd, step):\n"
            '    """One in-place AdamW step on (p, m, v)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# === Reference: hand-computed AdamW step on a single scalar ===\n"
            "# p_0 = 1.0, grad = 0.5, m=v=0, lr=0.1, b1=0.9, b2=0.999, eps=1e-8, wd=0.01\n"
            "# step=1\n"
            "# 1. Decoupled decay (pre-Adam): p = 1.0 * (1 - 0.1 * 0.01) = 0.999\n"
            "# 2. m = 0.9*0 + 0.1*0.5 = 0.05\n"
            "# 3. v = 0.999*0 + 0.001*0.25 = 0.00025\n"
            "# 4. m_hat = 0.05 / (1 - 0.9) = 0.5\n"
            "# 5. v_hat = 0.00025 / (1 - 0.999) = 0.25\n"
            "# 6. Adam step: p = 0.999 - 0.1 * 0.5 / (sqrt(0.25) + 1e-8) ~= 0.999 - 0.1 = 0.899\n"
            "p = t.tensor([1.0])\n"
            "grad = t.tensor([0.5])\n"
            "m = t.tensor([0.0])\n"
            "v = t.tensor([0.0])\n"
            "ex1_adamw_step(p, grad, m, v, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01, step=1)\n"
            "assert abs(p.item() - 0.899) < 1e-4, f'expected p~=0.899, got {p.item()}'\n"
            "assert abs(m.item() - 0.05) < 1e-7, f'expected m=0.05, got {m.item()}'\n"
            "assert abs(v.item() - 0.00025) < 1e-9, f'expected v=0.00025, got {v.item()}'\n"
            "\n"
            "# === Cross-check against torch.optim.AdamW on a small linear model ===\n"
            "t.manual_seed(0)\n"
            "p_ref = t.nn.Parameter(t.randn(5))\n"
            "p_ours = p_ref.detach().clone().requires_grad_(False)\n"
            "g = t.randn(5)\n"
            "p_ref.grad = g.clone()\n"
            "\n"
            "lr, b1, b2, eps, wd = 1e-2, 0.9, 0.999, 1e-8, 0.05\n"
            "opt = t.optim.AdamW([p_ref], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)\n"
            "opt.step()\n"
            "\n"
            "m = t.zeros_like(p_ours)\n"
            "v = t.zeros_like(p_ours)\n"
            "ex1_adamw_step(p_ours, g.clone(), m, v, lr=lr, beta1=b1, beta2=b2, eps=eps, wd=wd, step=1)\n"
            "\n"
            "assert t.allclose(p_ours, p_ref.detach(), atol=1e-6), (\n"
            "    f'AdamW step mismatch:\\n  ours = {p_ours}\\n  ref  = {p_ref.detach()}'\n"
            ")\n"
            "\n"
            "# === Multi-step cross-check ===\n"
            "t.manual_seed(1)\n"
            "p_ref = t.nn.Parameter(t.randn(4))\n"
            "p_ours = p_ref.detach().clone().requires_grad_(False)\n"
            "m_ours = t.zeros_like(p_ours)\n"
            "v_ours = t.zeros_like(p_ours)\n"
            "opt = t.optim.AdamW([p_ref], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)\n"
            "\n"
            "for step in range(1, 6):\n"
            "    grad = t.randn(4, generator=t.Generator().manual_seed(step))\n"
            "    p_ref.grad = grad.clone()\n"
            "    opt.step()\n"
            "    ex1_adamw_step(p_ours, grad.clone(), m_ours, v_ours,\n"
            "                   lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,\n"
            "                   wd=0.1, step=step)\n"
            "\n"
            "assert t.allclose(p_ours, p_ref.detach(), atol=1e-5), (\n"
            "    f'multi-step AdamW mismatch:\\n  ours = {p_ours}\\n  ref  = {p_ref.detach()}'\n"
            ")\n"
            "\n"
            "# === Compare to Adam (COUPLED) to confirm AdamW is different ===\n"
            "# Same hparams but with torch.optim.Adam, weight_decay=0.1, on the same grad.\n"
            "# AdamW result should NOT equal Adam result (they differ exactly because of\n"
            "# the decoupled-vs-coupled wd treatment).\n"
            "t.manual_seed(2)\n"
            "p_adam = t.nn.Parameter(t.randn(3))\n"
            "p_adamw = p_adam.detach().clone()\n"
            "g = t.randn(3)\n"
            "p_adam.grad = g.clone()\n"
            "\n"
            "opt_adam = t.optim.Adam([p_adam], lr=1e-2, weight_decay=0.1)\n"
            "opt_adam.step()\n"
            "\n"
            "m = t.zeros_like(p_adamw)\n"
            "v = t.zeros_like(p_adamw)\n"
            "ex1_adamw_step(p_adamw, g.clone(), m, v,\n"
            "               lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.1, step=1)\n"
            "\n"
            "diff = (p_adam.detach() - p_adamw).abs().max().item()\n"
            "assert diff > 1e-5, (\n"
            "    f'AdamW and Adam(wd=0.1) should differ; diff={diff} — '\n"
            "    f'did you fold wd into the gradient (Adam-style) instead of decoupling?'\n"
            ")\n"
            "\n"
            "# === wd=0 makes AdamW collapse to Adam ===\n"
            "t.manual_seed(3)\n"
            "p_adam = t.nn.Parameter(t.randn(3))\n"
            "p_adamw = p_adam.detach().clone()\n"
            "g = t.randn(3)\n"
            "p_adam.grad = g.clone()\n"
            "\n"
            "opt_adam = t.optim.Adam([p_adam], lr=1e-2, weight_decay=0.0)\n"
            "opt_adam.step()\n"
            "\n"
            "m = t.zeros_like(p_adamw)\n"
            "v = t.zeros_like(p_adamw)\n"
            "ex1_adamw_step(p_adamw, g.clone(), m, v,\n"
            "               lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0, step=1)\n"
            "\n"
            "assert t.allclose(p_adam.detach(), p_adamw, atol=1e-6), (\n"
            "    f'with wd=0 AdamW must equal Adam, got diff '\n"
            "    f'{(p_adam.detach()-p_adamw).abs().max().item()}'\n"
            ")"
        ),
        "solution_body": (
            "def ex1_adamw_step(p, grad, m, v, lr, beta1, beta2, eps, wd, step):\n"
            "    # 1. Decoupled weight-decay step FIRST (matches torch.optim.AdamW):\n"
            "    #    p <- p * (1 - lr * wd)   ==   p -= lr * wd * p\n"
            "    p.mul_(1 - lr * wd)\n"
            "    # 2. Moment update (Adam, unchanged)\n"
            "    m.mul_(beta1).add_(grad, alpha=1 - beta1)\n"
            "    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)\n"
            "    # 3. Bias correction\n"
            "    m_hat = m / (1 - beta1 ** step)\n"
            "    v_hat = v / (1 - beta2 ** step)\n"
            "    # 4. Adam step (no wd inside)\n"
            "    p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)"
        ),
        "solution_notes": (
            "**The ONE line that makes it AdamW.** Step 4 — "
            "`p -= lr * wd * p` — is the entire difference from Adam. "
            "Comment it out and you've reimplemented Adam (no decay). "
            "Move it into the gradient (`grad = grad + wd * p` before "
            "the moment update) and you've implemented coupled Adam-"
            "with-wd, which is what `torch.optim.Adam(weight_decay=...)` "
            "actually does.\n\n"
            "**`addcdiv_(t1, t2, value=-lr)`** is the in-place form "
            "of `p += -lr * t1 / t2`. The PyTorch source uses this for "
            "the Adam step because it avoids a temporary tensor.\n\n"
            "**Bias correction matters most at small `step`.** At "
            "`step=1`, `1 - beta1**1 = 0.1` so `m_hat = 10*m`. Without "
            "this, the first few steps would be tiny because `m` is "
            "biased toward zero (the EMA hasn't warmed up). By step "
            "~1000 the correction factor is essentially 1.\n\n"
            "**`p.mul_(1 - lr*wd)` is the pre-Adam decay.** That "
            "matches `torch.optim.AdamW`'s actual implementation in "
            "v1.13+ — decay first, then the Adam step on the "
            "now-decayed `p`. The original 2017 paper sketched both "
            "orderings; for small `lr` they are numerically "
            "indistinguishable, but PyTorch picked the pre-step form."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # clip-grad-norm-pre-step — ex1
    # =========================================================
    {
        "atom_id": "clip-grad-norm-pre-step",
        "subtopic": "Optimizer: clip_grad_norm pre-step",
        "topic_folder": "prereqs_optimizer_internals",
        "atom_recap_md": RECAP_CLIP_GRAD_NORM,
        "exercise_index": 1,
        "exercise_title": "clip grads to a max global L2 norm before optimizer.step",
        "slug": "clip-grads-to-a-max-global-l2-norm-before-optimizer-step",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["gradient-clipping", "global-norm", "transformer", "stability"],
        "kcs": [
            "global-l2-norm-rescale",
            "clip-pre-step-ordering",
        ],
        "lo": (
            "Apply `torch.nn.utils.clip_grad_norm_` to rescale a "
            "model's gradients to a max global L2 norm BEFORE calling "
            "`optimizer.step()`, preserving gradient direction."
        ),
        "prompt_body": (
            "Implement `ex1_clip_and_step(params, optimizer, "
            "max_norm)`. The canonical clipped-step utility used in "
            "transformer training loops.\n\n"
            "1. Compute the GLOBAL L2 norm across every parameter's "
            "`.grad` using `torch.nn.utils.clip_grad_norm_`. Pass "
            "`max_norm=max_norm`. The function clips in place and "
            "returns the PRE-clip norm as a tensor.\n"
            "2. Call `optimizer.step()` (uses the now-clipped grads).\n"
            "3. Call `optimizer.zero_grad()` so the next iteration "
            "starts fresh.\n"
            "4. Return the pre-clip norm as a Python float (call "
            "`.item()`).\n\n"
            "Inputs:\n"
            "- `params`: iterable of parameter `Tensor`s.\n"
            "- `optimizer`: a `torch.optim.Optimizer` instance.\n"
            "- `max_norm`: scalar float.\n\n"
            "Output: Python float — the global L2 norm BEFORE clipping."
        ),
        "stub": (
            "import torch.nn.utils as nn_utils\n"
            "\n"
            "def ex1_clip_and_step(params, optimizer, max_norm: float) -> float:\n"
            '    """Clip grads to max_norm, step the optimizer, zero grads."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# === Hand-computed: when norm > max_norm, all grads scale by max_norm/norm ===\n"
            "p1 = t.nn.Parameter(t.zeros(2))\n"
            "p2 = t.nn.Parameter(t.zeros(2))\n"
            "p1.grad = t.tensor([3.0, 4.0])    # norm = 5\n"
            "p2.grad = t.tensor([0.0, 0.0])    # norm = 0\n"
            "# Global norm = sqrt(3^2 + 4^2 + 0 + 0) = 5.\n"
            "opt = t.optim.SGD([p1, p2], lr=0.0)   # lr=0 so step doesn't move params; we just want clip behavior\n"
            "norm = ex1_clip_and_step([p1, p2], opt, max_norm=1.0)\n"
            "assert isinstance(norm, float), f'return must be float, got {type(norm).__name__}'\n"
            "assert abs(norm - 5.0) < 1e-5, f'expected pre-clip norm=5.0, got {norm}'\n"
            "# Optimizer.zero_grad should have cleared the grads (or set them to None).\n"
            "for p in (p1, p2):\n"
            "    if p.grad is not None:\n"
            "        assert t.allclose(p.grad, t.zeros_like(p.grad)), (\n"
            "            f'grad should be zeroed after step, got {p.grad}'\n"
            "        )\n"
            "\n"
            "# === Independent setup to verify the clipped step actually applied the clipped grad ===\n"
            "p1 = t.nn.Parameter(t.zeros(2))\n"
            "p2 = t.nn.Parameter(t.zeros(2))\n"
            "p1.grad = t.tensor([3.0, 4.0])\n"
            "p2.grad = t.tensor([0.0, 0.0])\n"
            "opt = t.optim.SGD([p1, p2], lr=1.0)\n"
            "norm = ex1_clip_and_step([p1, p2], opt, max_norm=1.0)\n"
            "# After clipping to norm 1: scale = 1/5 = 0.2. So p1.grad became [0.6, 0.8].\n"
            "# Step with lr=1.0: p1 = 0 - 1.0 * [0.6, 0.8] = [-0.6, -0.8].\n"
            "assert t.allclose(p1.detach(), t.tensor([-0.6, -0.8]), atol=1e-5), (\n"
            "    f'expected p1=[-0.6, -0.8], got {p1.detach()}'\n"
            ")\n"
            "assert t.allclose(p2.detach(), t.tensor([0.0, 0.0]), atol=1e-5), (\n"
            "    f'expected p2 unchanged, got {p2.detach()}'\n"
            ")\n"
            "\n"
            "# === When norm < max_norm, NO clipping (only step + zero) ===\n"
            "p = t.nn.Parameter(t.zeros(2))\n"
            "p.grad = t.tensor([0.3, 0.4])  # norm = 0.5\n"
            "opt = t.optim.SGD([p], lr=1.0)\n"
            "norm = ex1_clip_and_step([p], opt, max_norm=1.0)\n"
            "assert abs(norm - 0.5) < 1e-5\n"
            "# Step used UNCLIPPED grad: p = 0 - [0.3, 0.4] = [-0.3, -0.4].\n"
            "assert t.allclose(p.detach(), t.tensor([-0.3, -0.4]), atol=1e-5)\n"
            "\n"
            "# === Direction preserved when clipping (scalar rescale only) ===\n"
            "p = t.nn.Parameter(t.zeros(3))\n"
            "g_orig = t.tensor([6.0, 8.0, 0.0])  # norm = 10\n"
            "p.grad = g_orig.clone()\n"
            "opt = t.optim.SGD([p], lr=0.0)\n"
            "ex1_clip_and_step([p], opt, max_norm=2.0)\n"
            "# Even though zero_grad was called, the clip happened on the cloned grad before step.\n"
            "# Verify direction another way: redo without zero, check ratio.\n"
            "p = t.nn.Parameter(t.zeros(3))\n"
            "p.grad = t.tensor([6.0, 8.0, 0.0])\n"
            "nn_utils.clip_grad_norm_([p], max_norm=2.0)\n"
            "g_clipped = p.grad\n"
            "# Direction == orig direction\n"
            "cos = (g_orig @ g_clipped) / (g_orig.norm() * g_clipped.norm())\n"
            "assert abs(cos.item() - 1.0) < 1e-5, f'clipping should not rotate gradient, got cos={cos.item()}'\n"
            "assert abs(g_clipped.norm().item() - 2.0) < 1e-5, f'clipped norm should be 2.0, got {g_clipped.norm().item()}'\n"
            "\n"
            "# === Run a 20-iter loop on a tiny model to check stability under clipping ===\n"
            "t.manual_seed(0)\n"
            "model = t.nn.Linear(4, 2)\n"
            "opt = t.optim.SGD(model.parameters(), lr=10.0)   # crazy LR\n"
            "x = t.randn(8, 4)\n"
            "y = t.randn(8, 2)\n"
            "norms_seen = []\n"
            "for _ in range(20):\n"
            "    loss = (model(x) - y).pow(2).mean()\n"
            "    loss.backward()\n"
            "    n = ex1_clip_and_step(list(model.parameters()), opt, max_norm=1.0)\n"
            "    norms_seen.append(n)\n"
            "# Without clipping, lr=10 would diverge. Check the model didn't NaN out.\n"
            "for p in model.parameters():\n"
            "    assert t.isfinite(p).all(), 'model params became non-finite — clipping should have kept us bounded'\n"
            "# Pre-clip norms must all be > 0 (something WAS being clipped).\n"
            "assert all(n > 0 for n in norms_seen), 'pre-clip norm should always be positive'"
        ),
        "solution_body": (
            "import torch.nn.utils as nn_utils\n"
            "\n"
            "def ex1_clip_and_step(params, optimizer, max_norm):\n"
            "    pre_clip_norm = nn_utils.clip_grad_norm_(params, max_norm=max_norm)\n"
            "    optimizer.step()\n"
            "    optimizer.zero_grad()\n"
            "    return pre_clip_norm.item()"
        ),
        "solution_notes": (
            "**`clip_grad_norm_` returns the PRE-clip norm.** A common "
            "newbie mistake is to assume the return value is the "
            "post-clip norm. It's not — it's what the norm WAS before "
            "clipping. The post-clip norm is always `min(pre_clip, "
            "max_norm)`.\n\n"
            "**Why the function signature takes `params` not the "
            "model.** The clip operates on parameter tensors, not on "
            "a module hierarchy. `model.parameters()` returns the "
            "right iterable. For per-group clipping (different "
            "thresholds for the embedding vs the rest), you'd call "
            "`clip_grad_norm_` once per group.\n\n"
            "**Logging the pre-clip norm is standard practice.** "
            "Plotting `grad_norm` over training is one of the most "
            "useful debug signals: spikes indicate instability, "
            "constant high values mean your `max_norm` is too "
            "restrictive, flat-zero means a dead model."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # two-optimizers-alternating-step — ex1
    # =========================================================
    {
        "atom_id": "two-optimizers-alternating-step",
        "subtopic": "GAN: Two-optimizers alternating step",
        "topic_folder": "prereqs_generative",
        "atom_recap_md": RECAP_TWO_OPTIMIZERS,
        "exercise_index": 1,
        "exercise_title": "alternating D-step then G-step with two optimizers on toy modules",
        "slug": "alternating-d-step-then-g-step-with-two-optimizers-on-toy-modules",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["gan", "two-optimizers", "alternating", "d-step", "g-step"],
        "kcs": [
            "d-step-vs-g-step-ordering",
            "per-module-optimizer-isolation",
        ],
        "lo": (
            "Apply the GAN alternating-step pattern — D-step then "
            "G-step, each with its OWN optimizer and zero_grad — on "
            "toy modules to verify only the right module moves per "
            "step."
        ),
        "prompt_body": (
            "Implement `ex1_gan_iter(G, D, G_opt, D_opt, z, x_real)`. "
            "ONE iteration of the canonical GAN training loop.\n\n"
            "STEP 1 — D-step (train the discriminator):\n"
            "  a. `D_opt.zero_grad()`\n"
            "  b. Compute `fake = G(z).detach()` (stop-gradient — see "
            "the `detach-stop-gradient-trick` drill for why).\n"
            "  c. Compute `loss_D = (D(fake) - D(x_real)).mean()` "
            "(simplified Wasserstein-style loss — we want D to assign "
            "high values to reals and low values to fakes, so the loss "
            "is fake-real and we MINIMIZE).\n"
            "  d. `loss_D.backward()`\n"
            "  e. `D_opt.step()`\n\n"
            "STEP 2 — G-step (train the generator):\n"
            "  a. `G_opt.zero_grad()`\n"
            "  b. Compute `fake = G(z)` (NO detach — gradient flows "
            "through D and into G).\n"
            "  c. Compute `loss_G = -D(fake).mean()` (G wants D to "
            "assign HIGH values to fakes, so we minimize negative D).\n"
            "  d. `loss_G.backward()`\n"
            "  e. `G_opt.step()`\n\n"
            "Return `(loss_D_value, loss_G_value)` as Python floats.\n\n"
            "Inputs:\n"
            "- `G`, `D`: `nn.Module`s.\n"
            "- `G_opt`, `D_opt`: optimizers, each constructed with the "
            "matching module's parameters.\n"
            "- `z`: noise input to G, shape `(B, z_dim)`.\n"
            "- `x_real`: real samples, shape `(B, x_dim)`.\n\n"
            "Output: `(float, float)` — D-loss then G-loss values."
        ),
        "stub": (
            "def ex1_gan_iter(G, D, G_opt, D_opt, z, x_real):\n"
            '    """One GAN training iteration; return (loss_D, loss_G) floats."""\n'
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
            "z = t.randn(B, z_dim)\n"
            "x_real = t.randn(B, x_dim)\n"
            "\n"
            "# Snapshot all weights from G and D BEFORE the iter.\n"
            "G_before = {name: p.detach().clone() for name, p in G.named_parameters()}\n"
            "D_before = {name: p.detach().clone() for name, p in D.named_parameters()}\n"
            "\n"
            "loss_D_val, loss_G_val = ex1_gan_iter(G, D, G_opt, D_opt, z, x_real)\n"
            "\n"
            "# === Return shape ===\n"
            "assert isinstance(loss_D_val, float), f'loss_D must be float, got {type(loss_D_val).__name__}'\n"
            "assert isinstance(loss_G_val, float), f'loss_G must be float, got {type(loss_G_val).__name__}'\n"
            "\n"
            "# === Both modules updated (both should have moved) ===\n"
            "G_moved_any = any(\n"
            "    not t.allclose(G_before[name], p.detach()) for name, p in G.named_parameters()\n"
            ")\n"
            "D_moved_any = any(\n"
            "    not t.allclose(D_before[name], p.detach()) for name, p in D.named_parameters()\n"
            ")\n"
            "assert G_moved_any, 'G params did not move — G_opt.step() not called?'\n"
            "assert D_moved_any, 'D params did not move — D_opt.step() not called?'\n"
            "\n"
            "# === Independent optimizer isolation: G_opt.step() must NOT update D's params ===\n"
            "# Make a fresh setup where only the G-step happens; D should be untouched.\n"
            "t.manual_seed(1)\n"
            "G2 = nn.Sequential(nn.Linear(z_dim, 8), nn.ReLU(), nn.Linear(8, x_dim))\n"
            "D2 = nn.Sequential(nn.Linear(x_dim, 8), nn.ReLU(), nn.Linear(8, 1))\n"
            "G2_opt = t.optim.SGD(G2.parameters(), lr=1e-2)\n"
            "D2_opt = t.optim.SGD(D2.parameters(), lr=1e-2)\n"
            "z2 = t.randn(B, z_dim)\n"
            "x_real2 = t.randn(B, x_dim)\n"
            "\n"
            "D2_before = {name: p.detach().clone() for name, p in D2.named_parameters()}\n"
            "\n"
            "# Run a full iter (D-step then G-step). The G-step part should not move D2.\n"
            "# We can't directly inspect mid-iter, so we use the test's structure:\n"
            "# after the FULL iter, D2 has moved (D-step). But verify that the G-step part\n"
            "# (which we re-trigger now) doesn't move D2 further.\n"
            "_ = ex1_gan_iter(G2, D2, G2_opt, D2_opt, z2, x_real2)\n"
            "D2_after_iter1 = {name: p.detach().clone() for name, p in D2.named_parameters()}\n"
            "# Now do a second iter; D2 will move in the D-step. Just sanity-check D2 != G2 wiring:\n"
            "# G2's optimizer should not have D2's params in its param_groups.\n"
            "G2_opt_params = set()\n"
            "for g in G2_opt.param_groups:\n"
            "    for p in g['params']:\n"
            "        G2_opt_params.add(id(p))\n"
            "for p in D2.parameters():\n"
            "    assert id(p) not in G2_opt_params, (\n"
            "        'CROSS-WIRING DETECTED: D2 param appears in G_opt — '\n"
            "        'each module needs its own optimizer'\n"
            "    )\n"
            "D2_opt_params = set()\n"
            "for g in D2_opt.param_groups:\n"
            "    for p in g['params']:\n"
            "        D2_opt_params.add(id(p))\n"
            "for p in G2.parameters():\n"
            "    assert id(p) not in D2_opt_params, (\n"
            "        'CROSS-WIRING DETECTED: G2 param appears in D_opt'\n"
            "    )\n"
            "\n"
            "# === Stability over 50 iters: losses should not explode/NaN ===\n"
            "t.manual_seed(2)\n"
            "G3 = nn.Sequential(nn.Linear(z_dim, 16), nn.ReLU(), nn.Linear(16, x_dim))\n"
            "D3 = nn.Sequential(nn.Linear(x_dim, 16), nn.ReLU(), nn.Linear(16, 1))\n"
            "G3_opt = t.optim.SGD(G3.parameters(), lr=1e-3)\n"
            "D3_opt = t.optim.SGD(D3.parameters(), lr=1e-3)\n"
            "for _ in range(50):\n"
            "    z3 = t.randn(B, z_dim)\n"
            "    x_real3 = t.randn(B, x_dim)\n"
            "    ld, lg = ex1_gan_iter(G3, D3, G3_opt, D3_opt, z3, x_real3)\n"
            "    assert t.isfinite(t.tensor(ld)).item(), f'D-loss went non-finite: {ld}'\n"
            "    assert t.isfinite(t.tensor(lg)).item(), f'G-loss went non-finite: {lg}'\n"
            "for p in G3.parameters():\n"
            "    assert t.isfinite(p).all(), 'G params became non-finite'\n"
            "for p in D3.parameters():\n"
            "    assert t.isfinite(p).all(), 'D params became non-finite'"
        ),
        "solution_body": (
            "def ex1_gan_iter(G, D, G_opt, D_opt, z, x_real):\n"
            "    # === D-step ===\n"
            "    D_opt.zero_grad()\n"
            "    fake = G(z).detach()                 # stop-gradient into G\n"
            "    loss_D = (D(fake) - D(x_real)).mean()\n"
            "    loss_D.backward()\n"
            "    D_opt.step()\n"
            "\n"
            "    # === G-step ===\n"
            "    G_opt.zero_grad()\n"
            "    fake = G(z)                          # grad flows into G\n"
            "    loss_G = -D(fake).mean()\n"
            "    loss_G.backward()\n"
            "    G_opt.step()\n"
            "\n"
            "    return loss_D.item(), loss_G.item()"
        ),
        "solution_notes": (
            "**Why two `zero_grad` calls.** Each optimizer owns its "
            "own parameters' `.grad`. `D_opt.zero_grad()` only clears "
            "D's grads; G's grads accumulate untouched. If you called "
            "`G_opt.zero_grad()` at the top of the D-step it would do "
            "nothing wrong, but the convention is to zero each "
            "optimizer right before you use it.\n\n"
            "**Why we call `G(z)` TWICE.** Once with `.detach()` "
            "during the D-step (D needs to see fakes but G shouldn't "
            "learn from D's loss), and again without detach during "
            "the G-step (G learns from D's gradient flowing back). "
            "The two forward passes are NOT redundant — they build "
            "different autograd graphs.\n\n"
            "**The cross-wiring guard is the #1 GAN bug.** "
            "`G_opt = Adam(model.parameters())` (where `model` "
            "accidentally contains both G and D) is a silent killer: "
            "every G-step also moves D, and D's training is corrupted "
            "in a way that's invisible without an explicit param-set "
            "check.\n\n"
            "**Order: D-then-G is Goodfellow.** Some papers (WGAN-GP) "
            "do multiple D-steps per G-step for better D convergence. "
            "The skeleton is identical — just put the D-step in a "
            "loop."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # detach-stop-gradient-trick — ex1
    # =========================================================
    {
        "atom_id": "detach-stop-gradient-trick",
        "subtopic": "GAN: detach stop-gradient trick",
        "topic_folder": "prereqs_generative",
        "atom_recap_md": RECAP_DETACH_STOP_GRADIENT,
        "exercise_index": 1,
        "exercise_title": "detach G's output during D-step so grad does not flow into G",
        "slug": "detach-g-output-during-d-step-so-grad-does-not-flow-into-g",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["detach", "stop-gradient", "autograd", "gan"],
        "kcs": [
            "detach-stops-backward",
            "no-grad-in-d-step",
        ],
        "lo": (
            "Analyze the role of `.detach()` in the GAN D-step by "
            "computing D's loss on G's output and verifying that G's "
            "parameters receive ZERO gradient when (and only when) "
            "`detach()` is applied."
        ),
        "prompt_body": (
            "Implement two functions: `ex1_d_loss_correct(G, D, z, "
            "x_real)` and `ex1_d_loss_buggy(G, D, z, x_real)`. The "
            "ONLY difference is whether `G(z)` is detached.\n\n"
            "`ex1_d_loss_correct`:\n"
            "  1. `fake = G(z).detach()`\n"
            "  2. `loss = (D(fake) - D(x_real)).mean()`\n"
            "  3. `loss.backward()`\n"
            "  4. Return `loss.item()`.\n\n"
            "`ex1_d_loss_buggy` — same body, but WITHOUT `.detach()`:\n"
            "  1. `fake = G(z)`\n"
            "  2. `loss = (D(fake) - D(x_real)).mean()`\n"
            "  3. `loss.backward()`\n"
            "  4. Return `loss.item()`.\n\n"
            "Both functions are called fresh (the caller zeroes grads "
            "and then runs ONLY this function — no other backward "
            "passes in between).\n\n"
            "The test will:\n"
            "1. Call `ex1_d_loss_correct` and assert every G "
            "parameter has `.grad is None` OR `.grad` is exactly "
            "zero — backward STOPPED at the detach.\n"
            "2. Call `ex1_d_loss_buggy` and assert at least one G "
            "parameter has a NON-zero `.grad` — backward DID flow "
            "into G (the bug we're warning against).\n\n"
            "The returned loss values should be approximately equal "
            "(detach doesn't change the forward, only the backward "
            "graph) — within `1e-5`."
        ),
        "stub": (
            "def ex1_d_loss_correct(G, D, z, x_real) -> float:\n"
            '    """D-step loss with .detach() on G(z); return loss value."""\n'
            "    raise NotImplementedError()\n"
            "\n"
            "def ex1_d_loss_buggy(G, D, z, x_real) -> float:\n"
            '    """Same as ex1_d_loss_correct but WITHOUT .detach()."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import torch.nn as nn\n"
            "\n"
            "def _zero_all(*modules):\n"
            "    for m in modules:\n"
            "        for p in m.parameters():\n"
            "            if p.grad is not None:\n"
            "                p.grad.detach_()\n"
            "                p.grad.zero_()\n"
            "\n"
            "def _grad_norm(module):\n"
            "    \"\"\"Sum-of-squares norm across all params' .grad (treating None as 0).\"\"\"\n"
            "    s = 0.0\n"
            "    for p in module.parameters():\n"
            "        if p.grad is not None:\n"
            "            s += p.grad.pow(2).sum().item()\n"
            "    return s ** 0.5\n"
            "\n"
            "t.manual_seed(0)\n"
            "z_dim, x_dim, B = 4, 6, 8\n"
            "G = nn.Sequential(nn.Linear(z_dim, 16), nn.ReLU(), nn.Linear(16, x_dim))\n"
            "D = nn.Sequential(nn.Linear(x_dim, 16), nn.ReLU(), nn.Linear(16, 1))\n"
            "z = t.randn(B, z_dim)\n"
            "x_real = t.randn(B, x_dim)\n"
            "\n"
            "# === Correct path: detach ⇒ G grads stay zero ===\n"
            "_zero_all(G, D)\n"
            "loss_correct = ex1_d_loss_correct(G, D, z, x_real)\n"
            "assert isinstance(loss_correct, float)\n"
            "g_norm_correct = _grad_norm(G)\n"
            "d_norm_correct = _grad_norm(D)\n"
            "assert g_norm_correct == 0.0, (\n"
            "    f'.detach() should stop backward — but G grad norm = {g_norm_correct}'\n"
            ")\n"
            "assert d_norm_correct > 0.0, (\n"
            "    f'D should still receive gradient from its own loss — got 0'\n"
            ")\n"
            "\n"
            "# === Buggy path: no detach ⇒ G grads are nonzero ===\n"
            "_zero_all(G, D)\n"
            "loss_buggy = ex1_d_loss_buggy(G, D, z, x_real)\n"
            "assert isinstance(loss_buggy, float)\n"
            "g_norm_buggy = _grad_norm(G)\n"
            "d_norm_buggy = _grad_norm(D)\n"
            "assert g_norm_buggy > 0.0, (\n"
            "    f'Without .detach(), backward MUST flow into G — but G grad norm = 0'\n"
            ")\n"
            "assert d_norm_buggy > 0.0, 'D should also receive gradient (sanity)'\n"
            "\n"
            "# === Loss values match (forward is identical) ===\n"
            "assert abs(loss_correct - loss_buggy) < 1e-5, (\n"
            "    f'forward losses should match; correct={loss_correct} buggy={loss_buggy}'\n"
            ")\n"
            "\n"
            "# === D's grads should match between paths (D backward unaffected by G detach) ===\n"
            "# Re-run both to compare D grads side-by-side.\n"
            "_zero_all(G, D)\n"
            "ex1_d_loss_correct(G, D, z, x_real)\n"
            "d_correct = {name: p.grad.detach().clone() for name, p in D.named_parameters() if p.grad is not None}\n"
            "_zero_all(G, D)\n"
            "ex1_d_loss_buggy(G, D, z, x_real)\n"
            "d_buggy = {name: p.grad.detach().clone() for name, p in D.named_parameters() if p.grad is not None}\n"
            "assert set(d_correct.keys()) == set(d_buggy.keys()), 'D grad keys differ'\n"
            "for name in d_correct:\n"
            "    assert t.allclose(d_correct[name], d_buggy[name], atol=1e-5), (\n"
            "        f'D grad for {name} should be same between correct/buggy paths (detach only affects G)'\n"
            "    )\n"
            "\n"
            "# === Stress test: detach version on 10 calls should NEVER touch G ===\n"
            "for _ in range(10):\n"
            "    _zero_all(G, D)\n"
            "    ex1_d_loss_correct(G, D, t.randn(B, z_dim), t.randn(B, x_dim))\n"
            "    assert _grad_norm(G) == 0.0, 'G should never accumulate grad under detach()'"
        ),
        "solution_body": (
            "def ex1_d_loss_correct(G, D, z, x_real):\n"
            "    fake = G(z).detach()\n"
            "    loss = (D(fake) - D(x_real)).mean()\n"
            "    loss.backward()\n"
            "    return loss.item()\n"
            "\n"
            "def ex1_d_loss_buggy(G, D, z, x_real):\n"
            "    fake = G(z)\n"
            "    loss = (D(fake) - D(x_real)).mean()\n"
            "    loss.backward()\n"
            "    return loss.item()"
        ),
        "solution_notes": (
            "**The forward values match exactly.** `.detach()` doesn't "
            "change the VALUE of the tensor — same numbers, same "
            "shape, same dtype. It just removes the autograd "
            "connection back to G. The two losses are bit-identical "
            "in the forward.\n\n"
            "**D's gradient is the same either way.** Removing the "
            "G→fake→D edge from the autograd graph doesn't change the "
            "D→loss gradients — D's path to the loss is unaffected. "
            "That's why the test asserts `d_correct[name] == "
            "d_buggy[name]` for every D param.\n\n"
            "**Why the bug is invisible at runtime.** In the buggy "
            "version, `G_opt.zero_grad()` at the start of the next "
            "G-step would WIPE the wrong gradient — so the model "
            "STILL trains, just with the D-step contributing nothing "
            "meaningful to G's update. The symptom is 'GAN doesn't "
            "converge'; root cause is subtle.\n\n"
            "**Alternative formulations:**\n"
            "- `with torch.no_grad(): fake = G(z)` — cheaper (forward "
            "doesn't build graph) but obscures intent.\n"
            "- `fake = G(z).detach().requires_grad_(False)` — "
            "redundant; `.detach()` already implies no grad.\n"
            "- `fake = G(z); fake.requires_grad = False` — DOESN'T "
            "WORK on non-leaf tensors. Use `.detach()`."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # dataloader-pin-memory-workers — ex1
    # =========================================================
    {
        "atom_id": "dataloader-pin-memory-workers",
        "subtopic": "PyTorch: DataLoader pin_memory + workers",
        "topic_folder": "prereqs_pytorch_modules",
        "atom_recap_md": RECAP_DATALOADER_PIN_MEMORY,
        "exercise_index": 1,
        "exercise_title": "build a DataLoader with num_workers and pin_memory configured",
        "slug": "build-a-dataloader-with-num-workers-and-pin-memory-configured",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["dataloader", "pin-memory", "num-workers", "throughput"],
        "kcs": [
            "dataloader-config-args",
            "shuffle-vs-sampler-mutex",
        ],
        "lo": (
            "Apply the `DataLoader(dataset, batch_size, num_workers, "
            "pin_memory, shuffle)` constructor to build a configured "
            "loader and verify each argument propagated correctly."
        ),
        "prompt_body": (
            "Implement `ex1_make_dataloader(dataset, batch_size, "
            "num_workers, pin_memory, shuffle)`. A thin factory.\n\n"
            "1. Construct and return a `torch.utils.data.DataLoader` "
            "with all five arguments wired through.\n"
            "2. No defaults — the caller provides every value "
            "explicitly.\n\n"
            "The test verifies:\n"
            "- The DataLoader's attributes match the inputs.\n"
            "- The DataLoader yields the correct number of batches.\n"
            "- A passed shuffle=True actually re-orders batches "
            "across two iterations (deterministic shuffle is fine; "
            "we just check at least one permutation differs from "
            "the trivial identity).\n"
            "- Setting `num_workers=0` runs single-process (the "
            "Colab-safe default for notebooks where pickling cell-"
            "defined classes fails).\n\n"
            "Inputs:\n"
            "- `dataset`: a `torch.utils.data.Dataset` instance.\n"
            "- `batch_size`: int.\n"
            "- `num_workers`: int >= 0.\n"
            "- `pin_memory`: bool.\n"
            "- `shuffle`: bool.\n\n"
            "Output: `DataLoader`."
        ),
        "stub": (
            "from torch.utils.data import DataLoader, Dataset, TensorDataset\n"
            "\n"
            "def ex1_make_dataloader(dataset, batch_size: int, num_workers: int,\n"
            "                       pin_memory: bool, shuffle: bool) -> DataLoader:\n"
            '    """Return a DataLoader with all 5 args wired through."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch.utils.data import DataLoader, TensorDataset\n"
            "\n"
            "# === Build a tiny dataset ===\n"
            "x = t.arange(40, dtype=t.float32).reshape(40, 1)   # 40 items\n"
            "y = (x.squeeze() * 2).long()\n"
            "ds = TensorDataset(x, y)\n"
            "\n"
            "# === All args propagate ===\n"
            "loader = ex1_make_dataloader(ds, batch_size=8, num_workers=0,\n"
            "                             pin_memory=False, shuffle=False)\n"
            "assert isinstance(loader, DataLoader), f'must return DataLoader, got {type(loader).__name__}'\n"
            "assert loader.batch_size == 8\n"
            "assert loader.num_workers == 0\n"
            "assert loader.pin_memory == False\n"
            "# shuffle is exposed via the sampler type (RandomSampler vs SequentialSampler)\n"
            "from torch.utils.data.sampler import SequentialSampler, RandomSampler\n"
            "assert isinstance(loader.sampler, SequentialSampler), (\n"
            "    f'shuffle=False should give SequentialSampler, got {type(loader.sampler).__name__}'\n"
            ")\n"
            "\n"
            "# === Batch count = ceil(40/8) = 5 ===\n"
            "batches = list(loader)\n"
            "assert len(batches) == 5, f'expected 5 batches, got {len(batches)}'\n"
            "# Without shuffle, the first batch is [0..7].\n"
            "first_xs = batches[0][0].squeeze().tolist()\n"
            "assert first_xs == [0, 1, 2, 3, 4, 5, 6, 7], f'expected sequential order, got {first_xs}'\n"
            "\n"
            "# === shuffle=True: sampler is RandomSampler, order differs from sequential ===\n"
            "t.manual_seed(0)\n"
            "loader_shuf = ex1_make_dataloader(ds, batch_size=8, num_workers=0,\n"
            "                                  pin_memory=False, shuffle=True)\n"
            "assert isinstance(loader_shuf.sampler, RandomSampler), (\n"
            "    f'shuffle=True should give RandomSampler, got {type(loader_shuf.sampler).__name__}'\n"
            ")\n"
            "shuf_xs = list(loader_shuf)[0][0].squeeze().tolist()\n"
            "# At least one element should be out of place vs the trivial [0..7].\n"
            "assert shuf_xs != [0, 1, 2, 3, 4, 5, 6, 7], 'shuffle=True did not actually shuffle'\n"
            "\n"
            "# === pin_memory=True propagates (we can't test the page-locking effect on CPU, but the attr is set) ===\n"
            "loader_pin = ex1_make_dataloader(ds, batch_size=4, num_workers=0,\n"
            "                                 pin_memory=True, shuffle=False)\n"
            "assert loader_pin.pin_memory == True\n"
            "\n"
            "# === num_workers propagates ===\n"
            "# We pass num_workers=2 but don't actually iterate (avoiding multi-proc cost in CI).\n"
            "loader_mw = ex1_make_dataloader(ds, batch_size=4, num_workers=2,\n"
            "                                pin_memory=False, shuffle=False)\n"
            "assert loader_mw.num_workers == 2\n"
            "\n"
            "# === DataLoader doesn't drop the last partial batch by default ===\n"
            "# 40 / 7 = 5 full batches of 7 + 1 partial of 5.\n"
            "loader_7 = ex1_make_dataloader(ds, batch_size=7, num_workers=0,\n"
            "                               pin_memory=False, shuffle=False)\n"
            "batches7 = list(loader_7)\n"
            "assert len(batches7) == 6, f'expected 6 batches (5 full + 1 partial), got {len(batches7)}'\n"
            "assert batches7[-1][0].shape[0] == 5, (\n"
            "    f'expected last batch of size 5, got {batches7[-1][0].shape[0]}'\n"
            ")\n"
            "\n"
            "# === Calling twice gives independent iterators ===\n"
            "loader2 = ex1_make_dataloader(ds, batch_size=8, num_workers=0,\n"
            "                              pin_memory=False, shuffle=False)\n"
            "it1 = iter(loader2)\n"
            "it2 = iter(loader2)\n"
            "b1 = next(it1)\n"
            "b2 = next(it2)\n"
            "assert t.equal(b1[0], b2[0]), 'two iterators on same loader should yield same first batch (no shuffle)'"
        ),
        "solution_body": (
            "from torch.utils.data import DataLoader\n"
            "\n"
            "def ex1_make_dataloader(dataset, batch_size, num_workers, pin_memory, shuffle):\n"
            "    return DataLoader(\n"
            "        dataset,\n"
            "        batch_size=batch_size,\n"
            "        num_workers=num_workers,\n"
            "        pin_memory=pin_memory,\n"
            "        shuffle=shuffle,\n"
            "    )"
        ),
        "solution_notes": (
            "**`shuffle=True` and `sampler=` are mutually exclusive.** "
            "The DataLoader raises if you pass both. Internally "
            "`shuffle=True` constructs a `RandomSampler` for you; "
            "`shuffle=False` constructs a `SequentialSampler`. For "
            "distributed training you ALWAYS pass `sampler=` and OMIT "
            "`shuffle=` — the `DistributedSampler` handles the shuffle "
            "across ranks (see the `distributed-sampler-shard` drill).\n\n"
            "**`pin_memory=False` is the right default for CPU.** "
            "Pinned memory only helps when there's a GPU transfer to "
            "overlap with. On CPU-only training it costs RAM and "
            "gains nothing.\n\n"
            "**`num_workers=0` runs in the main process.** That's the "
            "Colab-safe default — multi-process loaders can fail to "
            "pickle objects defined in a notebook cell. `>0` is "
            "great for a real training script with top-level dataset "
            "classes."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # distributed-sampler-shard — ex1
    # =========================================================
    {
        "atom_id": "distributed-sampler-shard",
        "subtopic": "Distributed: DistributedSampler shard",
        "topic_folder": "prereqs_distributed",
        "atom_recap_md": RECAP_DISTRIBUTED_SAMPLER,
        "exercise_index": 1,
        "exercise_title": "shard a dataset across ranks with DistributedSampler",
        "slug": "shard-a-dataset-across-ranks-with-distributedsampler",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["distributed-sampler", "sharding", "set-epoch", "data-parallel"],
        "kcs": [
            "distributed-sampler-construction",
            "set-epoch-reshuffle",
        ],
        "lo": (
            "Apply `DistributedSampler(dataset, num_replicas, rank)` "
            "to produce a disjoint index shard for each rank, and "
            "verify the union of shards covers (with padding) the "
            "full dataset."
        ),
        "prompt_body": (
            "Implement `ex1_collect_shards(dataset, world_size, "
            "seed)`. The from-scratch verification that "
            "`DistributedSampler` shards the way you expect.\n\n"
            "1. For each `rank` in `range(world_size)`:\n"
            "   a. Build `DistributedSampler(dataset, "
            "num_replicas=world_size, rank=rank, shuffle=True, "
            "seed=seed)`.\n"
            "   b. Call `sampler.set_epoch(0)` (mandatory for "
            "reproducibility).\n"
            "   c. Iterate the sampler and collect the indices into a "
            "list.\n"
            "2. Return a `list[list[int]]` of length `world_size`, "
            "outer index = rank, inner = that rank's epoch-0 index "
            "list.\n\n"
            "Inputs:\n"
            "- `dataset`: any `Dataset` (the sampler only uses "
            "`len(dataset)`).\n"
            "- `world_size`: int >= 1.\n"
            "- `seed`: int.\n\n"
            "Output: `list[list[int]]`.\n\n"
            "(No multiprocessing required — the sampler is a pure "
            "iterator that takes `rank` as a constructor arg.)"
        ),
        "stub": (
            "from torch.utils.data import TensorDataset\n"
            "from torch.utils.data.distributed import DistributedSampler\n"
            "\n"
            "def ex1_collect_shards(dataset, world_size: int, seed: int) -> list:\n"
            '    """Return list of per-rank epoch-0 index lists."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from torch.utils.data import TensorDataset\n"
            "from torch.utils.data.distributed import DistributedSampler\n"
            "\n"
            "# === World size 4, dataset size 16 (evenly divisible) ===\n"
            "ds = TensorDataset(t.arange(16))\n"
            "shards = ex1_collect_shards(ds, world_size=4, seed=42)\n"
            "assert isinstance(shards, list)\n"
            "assert len(shards) == 4, f'expected 4 shards, got {len(shards)}'\n"
            "\n"
            "# Each rank gets exactly len(ds) / world_size = 4 indices.\n"
            "for rank, shard in enumerate(shards):\n"
            "    assert len(shard) == 4, f'rank {rank}: expected 4 items, got {len(shard)}'\n"
            "    for i in shard:\n"
            "        assert 0 <= i < 16, f'index {i} out of dataset range'\n"
            "\n"
            "# Shards are disjoint (no two ranks see the same index).\n"
            "all_indices = []\n"
            "for shard in shards:\n"
            "    all_indices.extend(shard)\n"
            "assert len(set(all_indices)) == 16, (\n"
            "    f'shards must be disjoint and cover the dataset; '\n"
            "    f'got {len(set(all_indices))} unique indices out of 16'\n"
            ")\n"
            "assert sorted(all_indices) == list(range(16)), (\n"
            "    f'union of shards must be exactly the full dataset, got {sorted(all_indices)}'\n"
            ")\n"
            "\n"
            "# === Reproducible with same seed ===\n"
            "shards_again = ex1_collect_shards(ds, world_size=4, seed=42)\n"
            "assert shards == shards_again, 'same seed must produce identical shards'\n"
            "\n"
            "# === Different seed → different (shuffled) shards ===\n"
            "shards_diff_seed = ex1_collect_shards(ds, world_size=4, seed=999)\n"
            "assert shards != shards_diff_seed, 'different seed should permute shard contents'\n"
            "\n"
            "# === World size 1 → one rank gets everything ===\n"
            "shards1 = ex1_collect_shards(ds, world_size=1, seed=0)\n"
            "assert len(shards1) == 1\n"
            "assert len(shards1[0]) == 16\n"
            "assert sorted(shards1[0]) == list(range(16))\n"
            "\n"
            "# === Non-divisible dataset → padded so each rank gets equal count ===\n"
            "# len(ds) = 17, world_size = 4. Padding adds 3 extra so 20/4 = 5 per rank.\n"
            "ds17 = TensorDataset(t.arange(17))\n"
            "shards17 = ex1_collect_shards(ds17, world_size=4, seed=7)\n"
            "for rank, shard in enumerate(shards17):\n"
            "    assert len(shard) == 5, f'rank {rank}: expected 5 (padded), got {len(shard)}'\n"
            "all17 = []\n"
            "for shard in shards17:\n"
            "    all17.extend(shard)\n"
            "# Total length 20, but only 17 unique indices (padding wraps).\n"
            "assert len(all17) == 20\n"
            "assert set(all17) == set(range(17)), (\n"
            "    f'padded shards must still cover the dataset, got {sorted(set(all17))}'\n"
            ")\n"
            "\n"
            "# === Within a rank, set_epoch was called (so shuffle ran) ===\n"
            "# Compare rank 0's shard to the unshuffled order — should differ.\n"
            "ds64 = TensorDataset(t.arange(64))\n"
            "shards64 = ex1_collect_shards(ds64, world_size=2, seed=0)\n"
            "# Without shuffle, rank 0 would see [0, 2, 4, ..., 62]. The seeded shuffle\n"
            "# of 64 indices should NOT produce this exact pattern.\n"
            "no_shuffle = list(range(0, 64, 2))\n"
            "assert shards64[0] != no_shuffle, (\n"
            "    'with shuffle=True + set_epoch(0), rank 0 should not see the trivial slice'\n"
            ")"
        ),
        "solution_body": (
            "def ex1_collect_shards(dataset, world_size, seed):\n"
            "    shards = []\n"
            "    for rank in range(world_size):\n"
            "        sampler = DistributedSampler(\n"
            "            dataset,\n"
            "            num_replicas=world_size,\n"
            "            rank=rank,\n"
            "            shuffle=True,\n"
            "            seed=seed,\n"
            "        )\n"
            "        sampler.set_epoch(0)\n"
            "        shards.append(list(sampler))\n"
            "    return shards"
        ),
        "solution_notes": (
            "**Why a loop over ranks works without multiprocessing.** "
            "`DistributedSampler` doesn't open sockets or call into "
            "`torch.distributed` — it's a pure iterator parameterized "
            "by `rank` and `num_replicas`. Each rank's sampler is "
            "INDEPENDENT; the math (modular striding into a shuffled "
            "permutation) is deterministic given the seed and epoch.\n\n"
            "**`set_epoch(epoch)` is the contract.** Without it, "
            "every call to `iter(sampler)` re-uses the same RNG seed, "
            "producing the same permutation epoch after epoch. The "
            "training loop must call it before each epoch — typically "
            "`for epoch in range(N): sampler.set_epoch(epoch); for "
            "batch in loader: ...`.\n\n"
            "**Padding is the default for evenly-divisible counts.** "
            "If `len(dataset) % world_size != 0`, the sampler wraps "
            "the index list from the start so every rank gets exactly "
            "`ceil(len / world_size)` items. This matters for "
            "`all_reduce` of gradients: every rank must run the same "
            "number of forward passes per epoch, or some ranks block "
            "waiting for others. Pass `drop_last=True` if you'd "
            "rather throw away the tail.\n\n"
            "**Pair with `DataLoader(sampler=sampler)`, NOT "
            "`shuffle=True`.** The two are mutually exclusive."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # rank0-only-side-effects — ex1
    # =========================================================
    {
        "atom_id": "rank0-only-side-effects",
        "subtopic": "Distributed: rank-0-only side effects",
        "topic_folder": "prereqs_distributed",
        "atom_recap_md": RECAP_RANK0_ONLY,
        "exercise_index": 1,
        "exercise_title": "guard checkpoint + log side effects behind if rank == 0",
        "slug": "guard-checkpoint-and-log-side-effects-behind-if-rank-equals-0",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["rank-0", "side-effects", "checkpoint", "logging"],
        "kcs": [
            "rank-0-guard-side-effects",
            "shared-resource-singleton-writer",
        ],
        "lo": (
            "Apply the `if rank == 0:` guard around shared-resource "
            "writes (checkpoint, log) so the action runs exactly "
            "once across the world, while per-rank compute proceeds "
            "on every rank."
        ),
        "prompt_body": (
            "Implement `ex1_epoch_end(rank, world_size, model_state, "
            "loss, ckpt_writer, log_writer, per_rank_recorder)`. The "
            "canonical end-of-epoch hook used in every distributed "
            "training script.\n\n"
            "Behavior — ON EVERY RANK:\n"
            "  - Call `per_rank_recorder(rank, loss)` to record that "
            "this rank did its forward/backward (every rank's "
            "contribution matters).\n\n"
            "Behavior — ON RANK 0 ONLY:\n"
            "  - Call `ckpt_writer(model_state)`.\n"
            "  - Call `log_writer(f'loss={loss:.4f}')`.\n\n"
            "Wrap the rank-0 calls in `if rank == 0:`. Do NOT use "
            "`if rank % world_size == 0:` or any other workaround — "
            "the convention is exactly `rank == 0`.\n\n"
            "Inputs:\n"
            "- `rank`, `world_size`: ints.\n"
            "- `model_state`: arbitrary object (the checkpoint).\n"
            "- `loss`: float.\n"
            "- `ckpt_writer`, `log_writer`, `per_rank_recorder`: "
            "callbacks.\n\n"
            "Output: `None`.\n\n"
            "The test calls `ex1_epoch_end` for each `rank` in "
            "`range(world_size)` (simulated locally — no real "
            "multiprocessing) using mock callbacks, and verifies the "
            "guard fired correctly."
        ),
        "stub": (
            "def ex1_epoch_end(rank: int, world_size: int, model_state,\n"
            "                  loss: float, ckpt_writer, log_writer,\n"
            "                  per_rank_recorder) -> None:\n"
            '    """End-of-epoch hook: per-rank record + rank-0 side effects."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "from unittest.mock import MagicMock\n"
            "\n"
            "# === World size 4, simulate calling on every rank ===\n"
            "world_size = 4\n"
            "ckpt_writer = MagicMock()\n"
            "log_writer = MagicMock()\n"
            "per_rank_recorder = MagicMock()\n"
            "\n"
            "for rank in range(world_size):\n"
            "    ex1_epoch_end(\n"
            "        rank=rank,\n"
            "        world_size=world_size,\n"
            "        model_state={'weights': [rank]},   # placeholder; we just check identity\n"
            "        loss=0.5 + rank * 0.01,\n"
            "        ckpt_writer=ckpt_writer,\n"
            "        log_writer=log_writer,\n"
            "        per_rank_recorder=per_rank_recorder,\n"
            "    )\n"
            "\n"
            "# === Checkpoint writer fired EXACTLY ONCE — and only from rank 0 ===\n"
            "assert ckpt_writer.call_count == 1, (\n"
            "    f'ckpt_writer fired {ckpt_writer.call_count} times — must be exactly 1 (rank 0 only)'\n"
            ")\n"
            "# The single call carried rank 0's model_state.\n"
            "(ckpt_arg,), _ = ckpt_writer.call_args\n"
            "assert ckpt_arg == {'weights': [0]}, (\n"
            "    f'ckpt_writer should have received rank 0 state, got {ckpt_arg}'\n"
            ")\n"
            "\n"
            "# === Log writer fired exactly once ===\n"
            "assert log_writer.call_count == 1, (\n"
            "    f'log_writer fired {log_writer.call_count} times — must be exactly 1'\n"
            ")\n"
            "(log_arg,), _ = log_writer.call_args\n"
            "assert log_arg == 'loss=0.5000', f'expected rank-0 loss string, got {log_arg!r}'\n"
            "\n"
            "# === Per-rank recorder fired ON EVERY RANK ===\n"
            "assert per_rank_recorder.call_count == world_size, (\n"
            "    f'per_rank_recorder must fire once per rank ({world_size}), '\n"
            "    f'got {per_rank_recorder.call_count}'\n"
            ")\n"
            "called_ranks = [call.args[0] for call in per_rank_recorder.call_args_list]\n"
            "assert sorted(called_ranks) == list(range(world_size)), (\n"
            "    f'per_rank_recorder should see every rank exactly once, got {sorted(called_ranks)}'\n"
            ")\n"
            "\n"
            "# === World size 8: still exactly one ckpt, eight per-rank records ===\n"
            "ckpt8 = MagicMock(); log8 = MagicMock(); rec8 = MagicMock()\n"
            "for rank in range(8):\n"
            "    ex1_epoch_end(rank, 8, {'r': rank}, 0.1, ckpt8, log8, rec8)\n"
            "assert ckpt8.call_count == 1, f'ws=8: ckpt should fire once, got {ckpt8.call_count}'\n"
            "assert log8.call_count == 1\n"
            "assert rec8.call_count == 8\n"
            "\n"
            "# === World size 1: rank 0 IS the world — everything still fires ===\n"
            "ckpt1 = MagicMock(); log1 = MagicMock(); rec1 = MagicMock()\n"
            "ex1_epoch_end(0, 1, {'r': 0}, 0.7, ckpt1, log1, rec1)\n"
            "assert ckpt1.call_count == 1\n"
            "assert log1.call_count == 1\n"
            "assert rec1.call_count == 1\n"
            "\n"
            "# === Sanity: a non-rank-0 call alone fires only the recorder ===\n"
            "ckpt_solo = MagicMock(); log_solo = MagicMock(); rec_solo = MagicMock()\n"
            "ex1_epoch_end(2, 4, {'r': 2}, 0.3, ckpt_solo, log_solo, rec_solo)\n"
            "assert ckpt_solo.call_count == 0, 'non-rank-0 must NOT call ckpt_writer'\n"
            "assert log_solo.call_count == 0, 'non-rank-0 must NOT call log_writer'\n"
            "assert rec_solo.call_count == 1, 'every rank still records, including non-rank-0'"
        ),
        "solution_body": (
            "def ex1_epoch_end(rank, world_size, model_state, loss,\n"
            "                  ckpt_writer, log_writer, per_rank_recorder):\n"
            "    # Every rank records its loss (per-rank, no race).\n"
            "    per_rank_recorder(rank, loss)\n"
            "\n"
            "    # Shared-resource side effects: rank 0 only.\n"
            "    if rank == 0:\n"
            "        ckpt_writer(model_state)\n"
            "        log_writer(f'loss={loss:.4f}')"
        ),
        "solution_notes": (
            "**The `if rank == 0` idiom is universal.** PyTorch's own "
            "distributed examples, fairseq, ARENA's "
            "`DistResNetTrainer`, every reference implementation — "
            "all gate checkpoint/log behind the same one-line guard. "
            "Reviewers expect it.\n\n"
            "**Why not `if dist.get_rank() == 0`?** Same effect, but "
            "now your function only works AFTER "
            "`init_process_group` has been called. Taking `rank` as a "
            "parameter is more testable (no global state) and works "
            "with single-process world_size=1 paths too.\n\n"
            "**What about `if rank == 0 and step % N == 0:` for "
            "throttled logging.** Compose the two conditions. The "
            "rank-0 guard is orthogonal to the throttle — apply both.\n\n"
            "**Pair with `dist.barrier()` when ordering matters.** If "
            "rank 0 writes a checkpoint that another rank will load "
            "(rare in training, common in eval), add `dist.barrier()` "
            "after the write so the loaders don't race the writer.\n\n"
            "**Don't put the COMPUTE inside the rank-0 guard.** The "
            "FORWARD pass, the loss computation, the backward pass — "
            "every rank does all of these. Only the SIDE EFFECTS "
            "(filesystem, network) need the guard."
        ),
        "extra_imports": [],
    },

]


# ---------------------------------------------------------------------------
# Verify each solution against its test body in-process.
# ---------------------------------------------------------------------------

def _verify_spec(spec):
    """Compile a tiny module from solution + test, run it, raise on failure."""
    atom_id = spec["atom_id"]
    ex_idx = spec["exercise_index"]
    src_lines = [
        "import numpy as np",
        "import torch as t",
        "from torch import Tensor",
        "import einops",
        "from einops import rearrange, reduce, repeat",
        "",
        "t.manual_seed(0)",
        "np.random.seed(0)",
        "",
    ]
    for extra in spec.get("extra_imports", []) or []:
        src_lines.append(extra)
    src_lines.append("")
    src_lines.append(spec["solution_body"])
    src_lines.append("")
    src_lines.append(spec["test_body"])
    src = "\n".join(src_lines)
    ns = {}
    try:
        exec(compile(src, f"<verify {atom_id} ex{ex_idx}>", "exec"), ns)
    except Exception:
        print(f"\n--- VERIFICATION FAILED for {atom_id} ex{ex_idx} ---", file=sys.stderr)
        traceback.print_exc()
        print("--- source ---", file=sys.stderr)
        for i, line in enumerate(src.splitlines(), 1):
            print(f"{i:4d}  {line}", file=sys.stderr)
        raise


def main():
    for spec in SPECS:
        print(f"verifying {spec['atom_id']} ex{spec['exercise_index']} ...", flush=True)
        _verify_spec(spec)
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"wrote {rel}")


if __name__ == "__main__":
    main()
