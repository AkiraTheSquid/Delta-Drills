#!/usr/bin/env python3
"""Author 8 ex2 deepening standalones for prereqs_adam_trainer atoms (batch H).

Each atom in prereqs_adam_trainer/ currently has ONE notebook (the ex1 ARENA
chap-0 / Adam-trainer prereq drill). This script emits a SECOND exercise per
atom that probes a DISTINCT facet not covered by ex1, following the PS4 /
Doughty contract:

  recap = facts + ONE worked exemplar (5-10 lines code) — no meta-commentary.
  exercise = stub + docstring + NotImplementedError + def _test_exN().
  visualization OK (matplotlib, headless via Agg backend).
  >=3 invariant assertions (shape/dtype/value/edge-cases).

Verify-before-emit: each spec is exec'd in a fresh namespace (stub then
solution_body then test_body) with `wandb` mocked. Fails fast if any
solution doesn't satisfy its own test.
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone


# ============================================================ ATOM 1 / 8
# argmax-accuracy-eval — ex2: weighted batch accumulation across loader
SPEC_ARGMAX_ACCURACY = {
    "atom_id": "argmax-accuracy-eval",
    "subtopic": "Eval: argmax accuracy",
    "topic_folder": "prereqs_adam_trainer",
    "atom_recap_md": (
        "## Weighted-average accuracy across a val loader — quick refresher\n"
        "\n"
        "A single-batch accuracy is `(logits.argmax(-1) == y).float().mean()`. "
        "Over a FULL val loader you can't just average per-batch accuracies "
        "naively: the last batch is often partial, and the unweighted mean "
        "over-counts it. The correct accumulator weights each batch by its "
        "size:\n"
        "\n"
        "```python\n"
        "correct, total = 0, 0\n"
        "with t.inference_mode():\n"
        "    for x, y in loader:\n"
        "        preds   = model(x).argmax(dim=-1)\n"
        "        correct += (preds == y).sum().item()\n"
        "        total   += y.shape[0]\n"
        "acc = correct / total          # weighted by per-batch size\n"
        "```\n"
        "\n"
        "Numerator counts integer hits; denominator counts examples. The "
        "ratio is mathematically identical to `mean of per-example correct` "
        "over the WHOLE eval set, regardless of how the loader batched it."
    ),
    "exercise_index": 2,
    "exercise_title": "weighted accuracy accumulator across a val loader (partial last batch)",
    "slug": "weighted-accuracy-accumulator-partial-last-batch",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["accuracy", "val-loader", "partial-batch", "weighted-average"],
    "kcs": ["accuracy-weighted-by-batch-size", "argmax-along-class-dim-minus-1"],
    "lo": (
        "Apply a `(correct, total)` integer accumulator over a val loader so "
        "the reported accuracy is exactly correct even when the last batch "
        "is partial."
    ),
    "prompt_body": (
        "Implement `ex2_eval_accuracy(model, loader)`. Run the model over "
        "the whole loader and return overall top-1 accuracy as a Python "
        "float in `[0, 1]`.\n\n"
        "1. Initialize integer counters `correct = 0`, `total = 0`.\n"
        "2. Under `t.inference_mode():`, iterate `(x, y) in loader`:\n"
        "   - `preds = model(x).argmax(dim=-1)` (shape matches `y`).\n"
        "   - `correct += (preds == y).sum().item()` — accumulate as int.\n"
        "   - `total   += y.shape[0]` — accumulate the per-batch size.\n"
        "3. Return `correct / total` (Python float).\n\n"
        "The test passes a loader whose LAST batch is partial (size 3 vs "
        "batch_size 8) and verifies the result matches the exact ratio "
        "computed by hand on the full dataset. A wrong implementation that "
        "averages per-batch accuracies fails this test because it weights "
        "the small last batch the same as the full ones."
    ),
    "stub": (
        "def ex2_eval_accuracy(model, loader) -> float:\n"
        '    """Weighted top-1 accuracy across a loader; handles partial last batch."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "from torch.utils.data import TensorDataset, DataLoader\n"
        "\n"
        "# === Construct a dataset where we KNOW the per-example correctness ===\n"
        "# 19 examples, batch_size=8 → batches of sizes [8, 8, 3] (partial last).\n"
        "B, C = 19, 4\n"
        "rng = t.Generator().manual_seed(0)\n"
        "X = t.randn(B, 6, generator=rng)\n"
        "Y = t.randint(0, C, (B,), generator=rng)\n"
        "ds = TensorDataset(X, Y)\n"
        "loader = DataLoader(ds, batch_size=8, shuffle=False)\n"
        "# Verify the loader actually emits a partial last batch.\n"
        "batch_sizes = [yy.shape[0] for _, yy in loader]\n"
        "assert batch_sizes == [8, 8, 3], f'unexpected batch sizes {batch_sizes}'\n"
        "\n"
        "model = t.nn.Linear(6, C)\n"
        "model.eval()\n"
        "\n"
        "# Ground truth: compute hits over the entire dataset directly.\n"
        "with t.inference_mode():\n"
        "    full_preds = model(X).argmax(dim=-1)\n"
        "ref_correct = int((full_preds == Y).sum().item())\n"
        "ref_acc = ref_correct / B\n"
        "\n"
        "acc = ex2_eval_accuracy(model, loader)\n"
        "assert isinstance(acc, float), f'must return float, got {type(acc).__name__}'\n"
        "assert 0.0 <= acc <= 1.0, f'accuracy out of [0,1]: {acc}'\n"
        "assert abs(acc - ref_acc) < 1e-6, (\n"
        "    f'weighted acc {acc} != reference {ref_acc}; partial-last-batch handling wrong'\n"
        ")\n"
        "\n"
        "# === Sanity: perfect-classifier case ===\n"
        "class _Perfect(t.nn.Module):\n"
        "    def __init__(self, n_classes):\n"
        "        super().__init__()\n"
        "        self.n_classes = n_classes\n"
        "    def forward(self, x):\n"
        "        # Use first column of x as the 'label hint' and produce confident logits.\n"
        "        labels = self._latest_labels\n"
        "        out = t.full((x.shape[0], self.n_classes), -10.0)\n"
        "        out[t.arange(x.shape[0]), labels] = 10.0\n"
        "        return out\n"
        "\n"
        "# Build a loader where the model has access to the labels (cheat-perfect).\n"
        "class _LabelLeakLoader:\n"
        "    def __init__(self, X, Y, batch_size):\n"
        "        self.X = X\n"
        "        self.Y = Y\n"
        "        self.batch_size = batch_size\n"
        "    def __iter__(self):\n"
        "        for i in range(0, len(self.X), self.batch_size):\n"
        "            xb = self.X[i:i+self.batch_size]\n"
        "            yb = self.Y[i:i+self.batch_size]\n"
        "            self._cur_y = yb\n"
        "            yield xb, yb\n"
        "\n"
        "perfect = _Perfect(C)\n"
        "leak = _LabelLeakLoader(X, Y, batch_size=8)\n"
        "# Hook labels into model right before forward by patching iterator.\n"
        "class _CheatModel(t.nn.Module):\n"
        "    def __init__(self, n_classes, X, Y):\n"
        "        super().__init__()\n"
        "        self.n_classes = n_classes\n"
        "        self.X = X\n"
        "        self.Y = Y\n"
        "    def forward(self, x):\n"
        "        # Match each row of x to its index in self.X to find its label.\n"
        "        out = t.full((x.shape[0], self.n_classes), -10.0)\n"
        "        for i, row in enumerate(x):\n"
        "            j = ((self.X - row).abs().sum(dim=-1) < 1e-6).nonzero(as_tuple=True)[0][0]\n"
        "            out[i, self.Y[j]] = 10.0\n"
        "        return out\n"
        "\n"
        "cheat = _CheatModel(C, X, Y)\n"
        "perfect_acc = ex2_eval_accuracy(cheat, loader)\n"
        "assert perfect_acc == 1.0, f'cheating-perfect model must give acc=1.0; got {perfect_acc}'\n"
        "\n"
        "# === Sanity: empty loader → would div-by-zero; test small but non-empty ===\n"
        "tiny_loader = DataLoader(TensorDataset(X[:1], Y[:1]), batch_size=1)\n"
        "tiny_acc = ex2_eval_accuracy(model, tiny_loader)\n"
        "assert tiny_acc in (0.0, 1.0), f'single-example loader gives 0 or 1; got {tiny_acc}'"
    ),
    "solution_body": (
        "def ex2_eval_accuracy(model, loader):\n"
        "    correct = 0\n"
        "    total = 0\n"
        "    with t.inference_mode():\n"
        "        for x, y in loader:\n"
        "            preds = model(x).argmax(dim=-1)\n"
        "            correct += (preds == y).sum().item()\n"
        "            total += y.shape[0]\n"
        "    return correct / total"
    ),
    "solution_notes": (
        "**Why integer accumulators, not float means.** "
        "`(correct / total)` is exact arithmetic over hit counts. A naive "
        "average-of-batch-means is biased whenever batches differ in size — "
        "the partial last batch is the canonical case. Same trick applies to "
        "weighted-average val loss (multiply per-batch loss by `x.shape[0]`).\n\n"
        "**Why `.item()` inside the loop.** Without it, you'd hold a "
        "growing list of 0-D tensors in autograd memory across the whole "
        "loader. `inference_mode` blocks autograd already, but `.item()` "
        "also detaches and releases the scalar — saves memory on large eval "
        "sets.\n\n"
        "**Top-K extension.** Replace `argmax(dim=-1)` with "
        "`topk(K, dim=-1).indices`, then `correct += (labels.unsqueeze(-1) "
        "== topk_idx).any(dim=-1).sum().item()`. Same accumulator pattern."
    ),
}


# ============================================================ ATOM 2 / 8
# bias-correction-divide — ex2: 10-step trajectory + visualization
SPEC_BIAS_CORRECTION = {
    "atom_id": "bias-correction-divide",
    "subtopic": "Optimizer: Adam bias-correction divide",
    "topic_folder": "prereqs_adam_trainer",
    "atom_recap_md": (
        "## Bias-correction trajectory across steps — quick refresher\n"
        "\n"
        "For constant gradient `g` the raw EMA converges to `g` from zero:\n"
        "```\n"
        "m_t = (1 - beta**t) * g          # raw EMA, biased toward 0\n"
        "m_hat_t = m_t / (1 - beta**t)    # = g exactly, for all t >= 1\n"
        "```\n"
        "The correction factor `1 / (1 - beta**t)` is HUGE early (e.g. "
        "step 1 with beta=0.9 → 10.0×) and fades to 1.0 as `t` grows. "
        "Plotted on a per-step axis, `m_hat` is a flat horizontal line at "
        "`g` from step 1 onwards while raw `m` only ramps in slowly.\n"
        "\n"
        "```python\n"
        "m = 0.0; beta = 0.9\n"
        "for tt in range(1, 11):\n"
        "    m = beta*m + (1-beta)*g                    # raw EMA\n"
        "    m_hat = m / (1 - beta**tt)                 # bias-corrected\n"
        "```"
    ),
    "exercise_index": 2,
    "exercise_title": "10-step bias-correction trajectory + convergence plot",
    "slug": "bias-correction-trajectory-10-steps-plot",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["bias-correction", "trajectory", "convergence", "visualization"],
    "kcs": ["bias-correction-divide-by-one-minus-beta-power-t", "ema-converges-to-constant-g"],
    "lo": (
        "Apply the EMA recurrence + bias-correction divide across 10 steps "
        "of a constant gradient to produce TWO trajectories — raw `m` "
        "(biased) and `m_hat` (unbiased, flat at `g`)."
    ),
    "prompt_body": (
        "Implement `ex2_bias_correction_trajectory(g, beta, n_steps)`. "
        "Run the Adam EMA + bias-correction for `n_steps` steps on a "
        "CONSTANT scalar gradient `g`.\n\n"
        "1. Start `m = 0.0` (Python float, scalar EMA buffer).\n"
        "2. For `step in range(1, n_steps + 1)`:\n"
        "   - `m = beta * m + (1 - beta) * g` — raw EMA update.\n"
        "   - `m_hat = m / (1 - beta ** step)` — bias-corrected.\n"
        "   - Append `m` to `raw_history`, `m_hat` to `corrected_history`.\n"
        "3. Return `(raw_history, corrected_history)` as two lists of "
        "floats, each of length `n_steps`.\n\n"
        "The test verifies:\n"
        "- Raw `m` ramps from `(1-beta)*g` toward `g`, never reaching it.\n"
        "- `m_hat` is EXACTLY `g` at every step (within float tolerance) "
        "for constant `g`.\n"
        "- A matplotlib plot of both trajectories is rendered (headless)."
    ),
    "stub": (
        "def ex2_bias_correction_trajectory(g: float, beta: float, n_steps: int) -> tuple:\n"
        '    """Return (raw_history, corrected_history) lists of floats, length n_steps."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "g, beta, n = 5.0, 0.9, 10\n"
        "raw, corr = ex2_bias_correction_trajectory(g, beta, n)\n"
        "\n"
        "# Shape / type invariants.\n"
        "assert isinstance(raw, list) and isinstance(corr, list)\n"
        "assert len(raw) == n and len(corr) == n, (\n"
        "    f'expected length {n}; got raw={len(raw)} corr={len(corr)}'\n"
        ")\n"
        "for v in raw + corr:\n"
        "    assert isinstance(v, float), f'history entries must be floats; got {type(v)}'\n"
        "\n"
        "# Bias-corrected: exactly g at every step (constant g case).\n"
        "for step, v in enumerate(corr, start=1):\n"
        "    assert abs(v - g) < 1e-6, (\n"
        "        f'corrected[step={step}] = {v} != g = {g}; '\n"
        "        f'bias correction failed for constant gradient'\n"
        "    )\n"
        "\n"
        "# Raw: closed form m_t = (1 - beta**t) * g.\n"
        "for step, v in enumerate(raw, start=1):\n"
        "    expected = (1 - beta ** step) * g\n"
        "    assert abs(v - expected) < 1e-6, (\n"
        "        f'raw[step={step}] = {v} != expected closed form {expected}'\n"
        "    )\n"
        "\n"
        "# Monotonic growth of raw toward g, never overshoots.\n"
        "assert all(raw[i] < raw[i+1] for i in range(n-1)), 'raw should grow monotonically'\n"
        "assert raw[-1] < g, f'raw should never reach g; got raw[-1]={raw[-1]} >= g={g}'\n"
        "assert raw[0] == (1 - beta) * g, f'step-1 raw should be (1-beta)*g'\n"
        "\n"
        "# Visualization: render to a figure (headless Agg backend).\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "steps = list(range(1, n + 1))\n"
        "ax.plot(steps, raw, marker='o', label='raw m_t (biased)')\n"
        "ax.plot(steps, corr, marker='s', label='m_hat_t (bias-corrected)')\n"
        "ax.axhline(g, linestyle='--', color='gray', label=f'true g={g}')\n"
        "ax.set_xlabel('step t')\n"
        "ax.set_ylabel('value')\n"
        "ax.set_title(f'Adam bias correction (beta={beta})')\n"
        "ax.legend()\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.close(fig)\n"
        "\n"
        "# Different beta sanity.\n"
        "raw2, corr2 = ex2_bias_correction_trajectory(2.0, 0.5, 5)\n"
        "for v in corr2:\n"
        "    assert abs(v - 2.0) < 1e-6, f'beta=0.5 corrected should still be 2.0; got {v}'\n"
        "assert raw2[0] == 0.5 * 2.0, f'beta=0.5 step-1 raw should be 1.0; got {raw2[0]}'"
    ),
    "solution_body": (
        "def ex2_bias_correction_trajectory(g, beta, n_steps):\n"
        "    m = 0.0\n"
        "    raw_history = []\n"
        "    corrected_history = []\n"
        "    for step in range(1, n_steps + 1):\n"
        "        m = beta * m + (1 - beta) * g\n"
        "        m_hat = m / (1 - beta ** step)\n"
        "        raw_history.append(float(m))\n"
        "        corrected_history.append(float(m_hat))\n"
        "    return raw_history, corrected_history"
    ),
    "solution_notes": (
        "**The plot is the insight.** Raw `m` ramps in slowly (still 65% "
        "of `g` at step 10 with beta=0.9); `m_hat` is flat at `g` from "
        "step 1. That flat line is what makes Adam usable from the first "
        "step — without correction, the warmup bias would silently slow "
        "training for ~100 steps with beta1=0.9.\n\n"
        "**Why scalar instead of tensor.** Same recurrence, no advantage "
        "to tensors here — the point is the trajectory math. A real Adam "
        "implementation does this per-element on the full Parameter "
        "tensor, but the per-step factor `(1 - beta**t)` is a scalar.\n\n"
        "**Generalization to non-constant g.** With a time-varying `g_t`, "
        "`m_hat_t` won't equal `g_t` exactly — it's a low-pass-filtered "
        "estimate. The bias correction still removes the zero-init bias, "
        "but the EMA filter still attenuates high-frequency content. "
        "That's the desired behavior."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
}


# ============================================================ ATOM 3 / 8
# cross-entropy-classification-loss — ex2: ignore_index (padding mask)
SPEC_CROSS_ENTROPY = {
    "atom_id": "cross-entropy-classification-loss",
    "subtopic": "Loss: Cross-entropy classification",
    "topic_folder": "prereqs_adam_trainer",
    "atom_recap_md": (
        "## Cross-entropy with `ignore_index` — quick refresher\n"
        "\n"
        "Language modeling and any sequence task pad short examples to a "
        "common length. The padding positions have a placeholder label "
        "(e.g. `-100` by convention) that the loss MUST skip — including "
        "them would let the model 'win' by predicting the pad token "
        "everywhere.\n"
        "\n"
        "`F.cross_entropy` has a built-in `ignore_index` argument that "
        "filters labels matching the sentinel BEFORE averaging:\n"
        "```python\n"
        "loss = F.cross_entropy(logits, labels, ignore_index=-100)\n"
        "# equivalent to:\n"
        "mask = labels != -100\n"
        "loss = F.cross_entropy(logits[mask], labels[mask])\n"
        "```\n"
        "The two forms are NUMERICALLY equivalent (mean over the kept "
        "examples only). The built-in arg saves the mask + index step."
    ),
    "exercise_index": 2,
    "exercise_title": "cross-entropy with ignore_index for masked padding labels",
    "slug": "cross-entropy-with-ignore-index-padding-mask",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["cross-entropy", "ignore-index", "padding-mask", "language-modeling"],
    "kcs": ["cross-entropy-takes-logits-not-probs", "cross-entropy-ignore-index-skips-padding"],
    "lo": (
        "Apply `ignore_index` filtering to cross-entropy so padded label "
        "positions don't contribute to the loss, and verify the masked "
        "result equals the masked-out hand-computed loss."
    ),
    "prompt_body": (
        "Implement `ex2_masked_cross_entropy(logits, labels, pad_id)`. "
        "The cross-entropy loss with padding positions masked out.\n\n"
        "Two acceptable implementations (both must give the same answer):\n"
        "1. **Built-in route**: `return F.cross_entropy(logits, labels, "
        "ignore_index=pad_id)`.\n"
        "2. **Manual route**: `mask = labels != pad_id; return "
        "F.cross_entropy(logits[mask], labels[mask])`.\n"
        "\n"
        "You can use either. The test verifies your output matches the "
        "manual reference EXACTLY (within 1e-5).\n\n"
        "Inputs:\n"
        "- `logits`: `(N, C)` float tensor.\n"
        "- `labels`: `(N,)` int64 — possibly containing `pad_id` "
        "(default -100) as a 'skip me' sentinel.\n"
        "- `pad_id`: int — labels equal to this are NOT scored.\n"
        "\n"
        "Output: scalar loss tensor.\n\n"
        "**Critical:** the loss must AVERAGE over the kept (non-padded) "
        "positions only, not over the total length. The test gives you a "
        "16-token batch with 6 padded positions; the correct denominator "
        "is 10, not 16."
    ),
    "stub": (
        "import torch.nn.functional as F\n"
        "\n"
        "\n"
        "def ex2_masked_cross_entropy(logits: Tensor, labels: Tensor, pad_id: int = -100) -> Tensor:\n"
        '    """Cross-entropy skipping positions where labels == pad_id."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn.functional as F\n"
        "\n"
        "# === 16 tokens, 6 padded ===\n"
        "rng = t.Generator().manual_seed(0)\n"
        "N, C = 16, 5\n"
        "logits = t.randn(N, C, generator=rng)\n"
        "labels = t.randint(0, C, (N,), generator=rng)\n"
        "# Pad 6 positions with -100.\n"
        "pad_positions = [1, 4, 7, 9, 12, 15]\n"
        "for p in pad_positions:\n"
        "    labels[p] = -100\n"
        "\n"
        "loss = ex2_masked_cross_entropy(logits, labels, pad_id=-100)\n"
        "\n"
        "# Reference 1: pytorch built-in.\n"
        "ref_builtin = F.cross_entropy(logits, labels, ignore_index=-100)\n"
        "# Reference 2: manual mask + index.\n"
        "mask = labels != -100\n"
        "ref_manual = F.cross_entropy(logits[mask], labels[mask])\n"
        "\n"
        "# All three must agree.\n"
        "assert isinstance(loss, t.Tensor), f'must return tensor, got {type(loss)}'\n"
        "assert loss.ndim == 0, f'must be scalar, got shape {tuple(loss.shape)}'\n"
        "assert t.allclose(loss, ref_builtin, atol=1e-5), (\n"
        "    f'mismatch vs built-in ignore_index: ours={loss.item():.6f} '\n"
        "    f'ref={ref_builtin.item():.6f}'\n"
        ")\n"
        "assert t.allclose(loss, ref_manual, atol=1e-5), (\n"
        "    f'mismatch vs manual mask: ours={loss.item():.6f} ref={ref_manual.item():.6f}'\n"
        ")\n"
        "assert t.allclose(ref_builtin, ref_manual, atol=1e-5), 'reference sanity'\n"
        "\n"
        "# === Denominator check: averaging over 10 kept positions, NOT 16 ===\n"
        "# Sum of per-kept-position losses divided by 10 must equal our loss.\n"
        "per_pos = F.cross_entropy(logits, labels, ignore_index=-100, reduction='none')\n"
        "# Padded positions emit 0 contribution.\n"
        "n_kept = int(mask.sum().item())\n"
        "assert n_kept == 10, f'mask sanity: expected 10 kept tokens, got {n_kept}'\n"
        "manual_mean = per_pos.sum() / n_kept\n"
        "assert t.allclose(loss, manual_mean, atol=1e-5), (\n"
        "    f'denominator should be n_kept=10, not N=16; got loss={loss.item():.6f} '\n"
        "    f'sum/10={manual_mean.item():.6f}'\n"
        ")\n"
        "\n"
        "# === Custom pad_id (not -100) ===\n"
        "labels2 = labels.clone()\n"
        "labels2[labels2 == -100] = 99       # repaint pad to 99\n"
        "loss2 = ex2_masked_cross_entropy(logits, labels2, pad_id=99)\n"
        "ref2 = F.cross_entropy(logits, labels2, ignore_index=99)\n"
        "assert t.allclose(loss2, ref2, atol=1e-5), f'custom pad_id mismatch'\n"
        "assert t.allclose(loss2, loss, atol=1e-5), 'pad_id is a label, sentinel value irrelevant'\n"
        "\n"
        "# === All-padded edge case: should be NaN (0/0); not asserted strictly ===\n"
        "labels_all_pad = t.full((4,), -100, dtype=t.long)\n"
        "logits_small = t.randn(4, C)\n"
        "loss_all_pad = ex2_masked_cross_entropy(logits_small, labels_all_pad, pad_id=-100)\n"
        "# PyTorch returns NaN for empty mean — we just verify it doesn't crash.\n"
        "assert loss_all_pad.shape == (), 'all-pad should still return a scalar'\n"
        "\n"
        "# === Grad still flows through unpadded positions ===\n"
        "logits_g = logits.detach().clone().requires_grad_(True)\n"
        "loss_g = ex2_masked_cross_entropy(logits_g, labels, pad_id=-100)\n"
        "loss_g.backward()\n"
        "assert logits_g.grad is not None\n"
        "# Padded rows MUST have zero grad.\n"
        "for p in pad_positions:\n"
        "    assert t.allclose(logits_g.grad[p], t.zeros(C)), (\n"
        "        f'pad row {p} got nonzero grad — pad should not contribute: {logits_g.grad[p]}'\n"
        "    )\n"
        "# Kept rows: at least one has nonzero grad.\n"
        "kept_rows = [i for i in range(N) if i not in pad_positions]\n"
        "assert any(logits_g.grad[i].abs().sum() > 0 for i in kept_rows), 'no grad on any kept row'"
    ),
    "solution_body": (
        "import torch.nn.functional as F\n"
        "\n"
        "\n"
        "def ex2_masked_cross_entropy(logits, labels, pad_id=-100):\n"
        "    return F.cross_entropy(logits, labels, ignore_index=pad_id)"
    ),
    "solution_notes": (
        "**Why -100 by convention.** PyTorch's `ignore_index` default is "
        "`-100` specifically because no valid class index is negative — "
        "you can't accidentally collide with a real label. HuggingFace "
        "tokenizers honor this convention: padded-token labels are set to "
        "-100 in their default data collators.\n\n"
        "**`ignore_index` vs `weight`.** `weight=t.tensor([...])` re-weights "
        "by class (e.g. for class imbalance); `ignore_index` removes "
        "specific POSITIONS entirely. Use both together for "
        "imbalanced-with-padding tasks like NER.\n\n"
        "**Where this lives in ARENA chap-3.** The transformer training "
        "loss is `F.cross_entropy(logits.view(-1, V), labels.view(-1), "
        "ignore_index=tokenizer.pad_token_id)` — exact same pattern, "
        "flattened across batch × seq."
    ),
}


# ============================================================ ATOM 4 / 8
# ema-first-moment — ex2: second moment v = beta2*v + (1-beta2)*g**2
SPEC_EMA_FIRST_MOMENT = {
    "atom_id": "ema-first-moment",
    "subtopic": "Optimizer: Adam EMA first moment",
    "topic_folder": "prereqs_adam_trainer",
    "atom_recap_md": (
        "## EMAs of `g` vs `g**2` side-by-side — quick refresher\n"
        "\n"
        "Adam carries TWO EMAs. ex1 covered the first moment "
        "`m = beta1*m + (1-beta1)*g`. The SECOND moment is structurally "
        "identical but EMAs `g**2`:\n"
        "```\n"
        "v_t = beta2 * v_{t-1} + (1 - beta2) * g_t**2\n"
        "```\n"
        "Key differences from `m`:\n"
        "- `v` is ALWAYS non-negative (squared input).\n"
        "- Default `beta2 = 0.999` (vs beta1=0.9) — longer averaging "
        "window (~1000 steps) because we want a stable variance estimate.\n"
        "- `v` does NOT preserve sign — it captures gradient MAGNITUDE only.\n"
        "\n"
        "```python\n"
        "for m_buf, v_buf, g in zip(m_list, v_list, grads):\n"
        "    m_buf.copy_(beta1 * m_buf + (1 - beta1) * g)\n"
        "    v_buf.copy_(beta2 * v_buf + (1 - beta2) * g.pow(2))\n"
        "```"
    ),
    "exercise_index": 2,
    "exercise_title": "second-moment EMA update v = beta2*v + (1-beta2)*g**2",
    "slug": "second-moment-ema-update-v-buffer-squared-gradient",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["adam", "second-moment", "ema", "squared-gradient"],
    "kcs": ["ema-second-moment-recurrence-squared-g", "buffer-copy_-mutates-state-in-place"],
    "lo": (
        "Apply the Adam second-moment recurrence `v = beta2*v + "
        "(1-beta2)*g**2` via `buffer.copy_()` so the variance-EMA buffer "
        "stays non-negative across steps regardless of gradient sign."
    ),
    "prompt_body": (
        "Implement `ex2_ema_v_step(v_list, grad_list, beta2)`. The "
        "second-moment update from Adam.\n\n"
        "For each `(v, g)` in `zip(v_list, grad_list)`:\n"
        "\n"
        "1. Compute `beta2 * v + (1 - beta2) * g.pow(2)`.\n"
        "2. Mutate `v` in place: `v.copy_(...)`. Don't rebind.\n"
        "3. Append `v` to the return list (by reference).\n"
        "\n"
        "Inputs:\n"
        "- `v_list`: list of per-param second-moment buffers (mutated).\n"
        "- `grad_list`: list of per-param gradients (NOT mutated).\n"
        "- `beta2`: float in `(0, 1)` — Adam default is `0.999`.\n"
        "\n"
        "Output: list of updated `v` tensors.\n\n"
        "**Critical invariant:** `v` must be NON-NEGATIVE elementwise at "
        "every step — the squaring guarantees this. The test verifies "
        "with deliberately-negative gradients."
    ),
    "stub": (
        "def ex2_ema_v_step(v_list: list, grad_list: list, beta2: float) -> list:\n"
        '    """In-place: v.copy_(beta2*v + (1-beta2)*g**2)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === One param, negative gradient — v must come out POSITIVE ===\n"
        "v = t.zeros(4)\n"
        "orig_id = id(v)\n"
        "orig_ptr = v.data_ptr()\n"
        "\n"
        "g = t.tensor([-1.0, -2.0, 3.0, -4.0])      # mixed signs\n"
        "beta2 = 0.999\n"
        "out = ex2_ema_v_step([v], [g], beta2=beta2)\n"
        "expected = (1 - beta2) * g.pow(2)           # all positive\n"
        "assert t.allclose(out[0], expected, atol=1e-9), (\n"
        "    f'step 1: expected {expected}, got {out[0]}'\n"
        ")\n"
        "assert t.allclose(v, expected, atol=1e-9), 'buffer not mutated in place'\n"
        "assert id(v) == orig_id, 'v was rebound — use v.copy_(...) not v = ...'\n"
        "assert v.data_ptr() == orig_ptr, 'storage reallocated — copy_ keeps storage'\n"
        "assert (v >= 0).all(), f'v must be non-negative even with negative g; got {v}'\n"
        "\n"
        "# === Multi-step: closed form for constant g ===\n"
        "# v_t = (1 - beta2^t) * g^2.\n"
        "v2 = t.zeros(3)\n"
        "g2 = t.tensor([2.0, -3.0, 4.0])\n"
        "beta2 = 0.99\n"
        "for step in range(1, 6):\n"
        "    ex2_ema_v_step([v2], [g2], beta2=beta2)\n"
        "    expected_step = (1 - beta2 ** step) * g2.pow(2)\n"
        "    assert t.allclose(v2, expected_step, atol=1e-6), (\n"
        "        f'step {step}: expected {expected_step}, got {v2}'\n"
        "    )\n"
        "    assert (v2 >= 0).all(), f'v must stay non-negative at step {step}'\n"
        "\n"
        "# === beta2 = 0 → v just equals g**2 ===\n"
        "v_fresh = t.zeros(3)\n"
        "ex2_ema_v_step([v_fresh], [t.tensor([2.0, -3.0, 4.0])], beta2=0.0)\n"
        "assert t.allclose(v_fresh, t.tensor([4.0, 9.0, 16.0])), (\n"
        "    f'beta2=0: v should equal g**2; got {v_fresh}'\n"
        ")\n"
        "\n"
        "# === Multi-param batch ===\n"
        "v_multi = [t.zeros(2), t.zeros(3, 3)]\n"
        "g_multi = [t.tensor([2.0, -2.0]), t.full((3, 3), -3.0)]\n"
        "ex2_ema_v_step(v_multi, g_multi, beta2=0.5)\n"
        "# 0.5 * 0 + 0.5 * [4, 4] = [2, 2]\n"
        "assert t.allclose(v_multi[0], t.tensor([2.0, 2.0]))\n"
        "# 0.5 * 0 + 0.5 * 9 = 4.5 everywhere\n"
        "assert t.allclose(v_multi[1], t.full((3, 3), 4.5))\n"
        "\n"
        "# === Input grad must not be mutated ===\n"
        "g_in = t.tensor([1.0, -2.0])\n"
        "g_snap = g_in.clone()\n"
        "ex2_ema_v_step([t.zeros(2)], [g_in], beta2=0.999)\n"
        "assert t.equal(g_in, g_snap), 'grad tensors must not be mutated'\n"
        "\n"
        "# === Sign-blindness contrast with first moment ===\n"
        "# Same magnitudes but flipped signs → identical v trajectory.\n"
        "g_pos = t.tensor([1.0, 2.0, 3.0])\n"
        "g_neg = -g_pos\n"
        "v_a, v_b = t.zeros(3), t.zeros(3)\n"
        "for _ in range(5):\n"
        "    ex2_ema_v_step([v_a], [g_pos], beta2=0.9)\n"
        "    ex2_ema_v_step([v_b], [g_neg], beta2=0.9)\n"
        "assert t.allclose(v_a, v_b), (\n"
        "    f'v should be sign-blind (squaring discards sign); got v_a={v_a}, v_b={v_b}'\n"
        ")"
    ),
    "solution_body": (
        "def ex2_ema_v_step(v_list, grad_list, beta2):\n"
        "    out = []\n"
        "    for v, g in zip(v_list, grad_list):\n"
        "        v.copy_(beta2 * v + (1 - beta2) * g.pow(2))\n"
        "        out.append(v)\n"
        "    return out"
    ),
    "solution_notes": (
        "**Why a separate drill for the second moment.** Structurally it's "
        "the first-moment update with `g.pow(2)` instead of `g` — same "
        "recurrence shape, same in-place semantics. But conflating the "
        "two (e.g. `v = beta1*v + ...` with the wrong beta, or forgetting "
        "to square) is the #2 source of Adam bugs after the rebind. "
        "Practicing the second-moment fold on its own builds the muscle "
        "memory for `g.pow(2)`.\n\n"
        "**`g.pow(2)` vs `g * g` vs `g ** 2`.** All three are equivalent; "
        "`g.pow(2)` is what PyTorch's reference Adam uses (it's a single "
        "dispatched op rather than two). For autograd this matters very "
        "little; for clarity the choice is taste.\n\n"
        "**Connecting to Adam's update.** The full Adam step is "
        "`theta -= lr * m_hat / (sqrt(v_hat) + eps)`. The `sqrt(v_hat)` "
        "is why `v` must stay non-negative — `sqrt` of a negative would "
        "be NaN, and the optimizer would silently produce all-NaN "
        "parameters within one step."
    ),
}


# ============================================================ ATOM 5 / 8
# examples-seen-step-axis — ex2: wandb.log with examples_seen key
SPEC_EXAMPLES_SEEN = {
    "atom_id": "examples-seen-step-axis",
    "subtopic": "Trainer: examples-seen step axis",
    "topic_folder": "prereqs_adam_trainer",
    "atom_recap_md": (
        "## Logging `examples_seen` to wandb — quick refresher\n"
        "\n"
        "The fair x-axis across runs with different batch sizes is "
        "`examples_seen = step * batch_size`. To make wandb actually USE "
        "that axis, you include it as a key in every `wandb.log(...)` "
        "call alongside the metrics — wandb's UI then lets you select it "
        "from the x-axis dropdown.\n"
        "\n"
        "```python\n"
        "for step, (x, y) in enumerate(loader, start=1):\n"
        "    loss = train_one_batch(x, y)\n"
        "    wandb.log({\n"
        "        'loss':          loss.item(),\n"
        "        'examples_seen': step * batch_size,\n"
        "    })\n"
        "```\n"
        "\n"
        "Every payload contains the same `examples_seen` key so wandb "
        "knows which metric is the proposed x-axis. The `step=...` kwarg "
        "is OPTIONAL — by default wandb auto-increments its internal step "
        "with each call."
    ),
    "exercise_index": 2,
    "exercise_title": "wandb.log payloads include examples_seen for cross-batch-size comparability",
    "slug": "wandb-log-payloads-include-examples-seen",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["wandb", "examples-seen", "logging", "x-axis"],
    "kcs": ["examples-seen-equals-step-times-batch-size", "wandb-log-payload-contains-examples-seen-key"],
    "lo": (
        "Apply `wandb.log({'loss': ..., 'examples_seen': step * "
        "batch_size})` inside a training loop so every payload carries "
        "the fair cross-batch-size x-axis."
    ),
    "prompt_body": (
        "Implement `ex2_train_with_wandb(losses, batch_size, wandb)`. A "
        "training-loop-like driver that emits one `wandb.log(...)` call "
        "per step, including the `examples_seen` key.\n\n"
        "1. For `step, loss in enumerate(losses, start=1)`:\n"
        "   - Compute `examples_seen = step * batch_size`.\n"
        "   - Call `wandb.log({'loss': loss, 'examples_seen': "
        "examples_seen})`.\n"
        "2. Return the total number of log calls made.\n"
        "\n"
        "Inputs:\n"
        "- `losses`: list of per-step float losses (the trainer's per-batch "
        "loss history).\n"
        "- `batch_size`: int — examples per batch.\n"
        "- `wandb`: a wandb-like object exposing `wandb.log(payload)`. The "
        "test passes a `MagicMock` instead of the real wandb.\n"
        "\n"
        "Output: int — the number of log calls (= len(losses)).\n\n"
        "**Critical:** EVERY payload must contain BOTH keys. The test "
        "inspects the mock's call args and asserts both `'loss'` and "
        "`'examples_seen'` appear in every dict passed to `wandb.log`."
    ),
    "stub": (
        "def ex2_train_with_wandb(losses: list, batch_size: int, wandb) -> int:\n"
        '    """Emit one wandb.log per step; each payload has loss + examples_seen."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "from unittest.mock import MagicMock\n"
        "\n"
        "# === Build a fake wandb ===\n"
        "wandb = MagicMock()\n"
        "\n"
        "# === 5-step run, batch_size=32 ===\n"
        "losses = [1.0, 0.8, 0.7, 0.65, 0.6]\n"
        "n_calls = ex2_train_with_wandb(losses, batch_size=32, wandb=wandb)\n"
        "\n"
        "assert n_calls == 5, f'expected 5 log calls, got {n_calls}'\n"
        "assert wandb.log.call_count == 5, (\n"
        "    f'wandb.log should be called once per step; got {wandb.log.call_count}'\n"
        ")\n"
        "\n"
        "# === Inspect each call's payload ===\n"
        "for i, call in enumerate(wandb.log.call_args_list, start=1):\n"
        "    args, kwargs = call\n"
        "    # Payload must be the first positional arg (a dict).\n"
        "    assert len(args) == 1, f'call {i}: expected one positional dict arg, got {args}'\n"
        "    payload = args[0]\n"
        "    assert isinstance(payload, dict), f'call {i}: payload must be dict, got {type(payload)}'\n"
        "    assert 'loss' in payload, f'call {i}: payload missing loss key: {payload}'\n"
        "    assert 'examples_seen' in payload, (\n"
        "        f'call {i}: payload missing examples_seen key: {payload}'\n"
        "    )\n"
        "    # examples_seen must equal step * 32.\n"
        "    assert payload['examples_seen'] == i * 32, (\n"
        "        f'call {i}: examples_seen should be {i*32}, got {payload[\"examples_seen\"]}'\n"
        "    )\n"
        "    # loss must match the input.\n"
        "    assert payload['loss'] == losses[i - 1], (\n"
        "        f'call {i}: loss should be {losses[i-1]}, got {payload[\"loss\"]}'\n"
        "    )\n"
        "\n"
        "# === Different batch size → different examples_seen scale ===\n"
        "wandb2 = MagicMock()\n"
        "ex2_train_with_wandb([0.5, 0.5], batch_size=128, wandb=wandb2)\n"
        "payloads2 = [c.args[0] for c in wandb2.log.call_args_list]\n"
        "assert payloads2[0]['examples_seen'] == 128\n"
        "assert payloads2[1]['examples_seen'] == 256, (\n"
        "    f'step 2 with batch 128 should be 256; got {payloads2[1][\"examples_seen\"]}'\n"
        ")\n"
        "\n"
        "# === Empty losses → zero calls ===\n"
        "wandb3 = MagicMock()\n"
        "n3 = ex2_train_with_wandb([], batch_size=64, wandb=wandb3)\n"
        "assert n3 == 0\n"
        "assert wandb3.log.call_count == 0\n"
        "\n"
        "# === Final-step cross-run comparability sanity ===\n"
        "# Two runs differing only in batch size should reach SAME examples_seen.\n"
        "wandb_a = MagicMock()\n"
        "wandb_b = MagicMock()\n"
        "ex2_train_with_wandb([0.1] * 10, batch_size=32, wandb=wandb_a)\n"
        "ex2_train_with_wandb([0.1] * 5,  batch_size=64, wandb=wandb_b)\n"
        "last_a = wandb_a.log.call_args_list[-1].args[0]['examples_seen']\n"
        "last_b = wandb_b.log.call_args_list[-1].args[0]['examples_seen']\n"
        "assert last_a == last_b == 320, (\n"
        "    f'run A (10*32) and run B (5*64) should both end at 320; got {last_a} / {last_b}'\n"
        ")"
    ),
    "solution_body": (
        "def ex2_train_with_wandb(losses, batch_size, wandb):\n"
        "    n = 0\n"
        "    for step, loss in enumerate(losses, start=1):\n"
        "        examples_seen = step * batch_size\n"
        "        wandb.log({'loss': loss, 'examples_seen': examples_seen})\n"
        "        n += 1\n"
        "    return n"
    ),
    "solution_notes": (
        "**Why include `examples_seen` in EVERY payload (not just "
        "occasionally).** Wandb's x-axis selector picks one key globally "
        "and applies it to all metrics. If some payloads have "
        "`examples_seen` and others don't, the curve gets gaps. Easiest "
        "rule: include it in every `wandb.log` call alongside whatever "
        "you're actually logging.\n\n"
        "**Why pass `wandb` as a parameter instead of `import wandb`.** "
        "Testability. The drill mocks `wandb`; a real trainer passes the "
        "live module. Same code path either way. ARENA's trainers wire "
        "wandb through the constructor for the same reason — easier to "
        "swap in a no-op logger for fast smoke tests.\n\n"
        "**`step=` kwarg.** Wandb's `log` accepts a `step=` integer to "
        "override its internal step counter. Useful when YOUR step "
        "counter and wandb's drift apart (e.g. you log multiple metrics "
        "between optimizer steps). For most loops you can omit it."
    ),
}


# ============================================================ ATOM 6 / 8
# optimizer-loop-on-tensor — ex2: SGD with momentum (adds buffer to the loop)
SPEC_OPTIMIZER_LOOP = {
    "atom_id": "optimizer-loop-on-tensor",
    "subtopic": "Optimizer: optimizer.step loop over params",
    "topic_folder": "prereqs_adam_trainer",
    "atom_recap_md": (
        "## SGD with momentum — explicit per-param loop + per-param buffer "
        "— quick refresher\n"
        "\n"
        "Plain SGD's `p -= lr * p.grad` becomes SGD-with-momentum by "
        "introducing a per-parameter velocity buffer:\n"
        "```\n"
        "for p, b in zip(self.params, self.buffers):\n"
        "    if p.grad is None:\n"
        "        continue\n"
        "    b.copy_(mu * b + p.grad)            # accumulate velocity in place\n"
        "    p -= self.lr * b                     # step along velocity\n"
        "```\n"
        "Three differences from plain SGD:\n"
        "1. Each parameter gets a `zeros_like(p)` buffer at init.\n"
        "2. The loop carries one extra `copy_` per param (velocity "
        "recurrence).\n"
        "3. The update reads from `b`, not directly from `p.grad`.\n"
        "\n"
        "Default momentum `mu = 0.9` matches PyTorch's `torch.optim.SGD` "
        "default. The `None`-grad guard and `@t.inference_mode()` "
        "decorator carry over unchanged from plain SGD."
    ),
    "exercise_index": 2,
    "exercise_title": "SGD with momentum: extend the per-param loop with a velocity buffer",
    "slug": "sgd-with-momentum-per-param-velocity-buffer",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["sgd-momentum", "velocity-buffer", "per-param-loop", "in-place"],
    "kcs": ["optimizer-step-explicit-for-loop-over-params", "per-param-state-buffer-allocation"],
    "lo": (
        "Apply the per-param explicit loop pattern to SGD-with-momentum, "
        "carrying a parallel velocity buffer alongside `self.params` and "
        "updating both in place under `@t.inference_mode()`."
    ),
    "prompt_body": (
        "Implement `Ex2MomentumSGD`. A hand-rolled SGD with momentum.\n\n"
        "1. `__init__(self, params, lr, momentum=0.9)`:\n"
        "   - Materialize `self.params = list(params)`.\n"
        "   - Store `self.lr = lr`, `self.momentum = momentum`.\n"
        "   - Allocate `self.bufs = [t.zeros_like(p) for p in self.params]`.\n"
        "2. `step(self)` decorated with `@t.inference_mode()`. For each "
        "`(p, b)` in `zip(self.params, self.bufs)`:\n"
        "   - If `p.grad is None`, SKIP.\n"
        "   - Else: `b.copy_(self.momentum * b + p.grad)`, then "
        "`p -= self.lr * b`.\n"
        "3. `zero_grad(self)`: `p.grad = None` for every param.\n"
        "\n"
        "The test compares your optimizer to PyTorch's `torch.optim.SGD"
        "(..., momentum=0.9)` over 10 steps on the same model + data and "
        "asserts the parameter trajectories are identical (within "
        "1e-5 atol). It also verifies the buffer is mutated IN PLACE "
        "(same `id` / `data_ptr` across steps)."
    ),
    "stub": (
        "class Ex2MomentumSGD:\n"
        "    \"\"\"Hand-rolled SGD with momentum — per-param loop + velocity buffer.\"\"\"\n"
        "\n"
        "    def __init__(self, params, lr: float, momentum: float = 0.9):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    def step(self):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    def zero_grad(self):\n"
        "        raise NotImplementedError()"
    ),
    "test_body": (
        "# === Single-param sanity ===\n"
        "p = t.nn.Parameter(t.tensor([10.0, 20.0, 30.0]))\n"
        "p.grad = t.tensor([1.0, 1.0, 1.0])\n"
        "opt = Ex2MomentumSGD([p], lr=0.1, momentum=0.9)\n"
        "\n"
        "# Buffer init checks.\n"
        "assert isinstance(opt.bufs, list) and len(opt.bufs) == 1\n"
        "assert opt.bufs[0].shape == p.shape\n"
        "assert t.all(opt.bufs[0] == 0)\n"
        "buf_id_before = id(opt.bufs[0])\n"
        "buf_ptr_before = opt.bufs[0].data_ptr()\n"
        "\n"
        "# Step 1: b = 0.9*0 + g = [1,1,1]; p -= 0.1 * [1,1,1] = -0.1 each.\n"
        "opt.step()\n"
        "assert t.allclose(p.detach(), t.tensor([9.9, 19.9, 29.9])), p.detach()\n"
        "assert t.allclose(opt.bufs[0], t.tensor([1.0, 1.0, 1.0]))\n"
        "assert id(opt.bufs[0]) == buf_id_before, 'buf must not be rebound'\n"
        "assert opt.bufs[0].data_ptr() == buf_ptr_before, 'buf storage must not be reallocated'\n"
        "\n"
        "# Step 2: same grad, b = 0.9 * 1 + 1 = 1.9; p -= 0.1 * 1.9 = -0.19.\n"
        "p.grad = t.tensor([1.0, 1.0, 1.0])\n"
        "opt.step()\n"
        "assert t.allclose(opt.bufs[0], t.tensor([1.9, 1.9, 1.9]))\n"
        "assert t.allclose(p.detach(), t.tensor([9.71, 19.71, 29.71]), atol=1e-6), p.detach()\n"
        "\n"
        "# === None-grad guard ===\n"
        "p_frozen = t.nn.Parameter(t.tensor([7.0, 7.0]))\n"
        "p_active = t.nn.Parameter(t.tensor([1.0, 2.0]))\n"
        "p_active.grad = t.tensor([10.0, 10.0])\n"
        "assert p_frozen.grad is None\n"
        "opt2 = Ex2MomentumSGD([p_frozen, p_active], lr=0.01, momentum=0.5)\n"
        "opt2.step()         # must not raise on None-grad\n"
        "# p_frozen should be untouched AND its buf should still be zero.\n"
        "assert t.allclose(p_frozen.detach(), t.tensor([7.0, 7.0]))\n"
        "assert t.all(opt2.bufs[0] == 0), f'frozen param buf should stay zero; got {opt2.bufs[0]}'\n"
        "# p_active moved.\n"
        "expected_buf_a = 0.5 * 0 + t.tensor([10.0, 10.0])\n"
        "assert t.allclose(opt2.bufs[1], expected_buf_a)\n"
        "assert t.allclose(p_active.detach(), t.tensor([0.9, 1.9]), atol=1e-6)\n"
        "\n"
        "# === zero_grad clears all grads ===\n"
        "opt2.zero_grad()\n"
        "for q in opt2.params:\n"
        "    assert q.grad is None\n"
        "\n"
        "# === Match torch.optim.SGD with momentum over 10 steps ===\n"
        "t.manual_seed(0)\n"
        "ref_model = t.nn.Linear(3, 1)\n"
        "our_model = t.nn.Linear(3, 1)\n"
        "# Copy initial weights from ref to ours (so they start identical).\n"
        "with t.no_grad():\n"
        "    our_model.weight.copy_(ref_model.weight)\n"
        "    our_model.bias.copy_(ref_model.bias)\n"
        "\n"
        "ref_opt = t.optim.SGD(ref_model.parameters(), lr=0.05, momentum=0.9)\n"
        "our_opt = Ex2MomentumSGD(our_model.parameters(), lr=0.05, momentum=0.9)\n"
        "\n"
        "X = t.randn(16, 3, generator=t.Generator().manual_seed(7))\n"
        "Y = t.randn(16, 1, generator=t.Generator().manual_seed(8))\n"
        "for step_i in range(10):\n"
        "    # Reference step.\n"
        "    ref_opt.zero_grad()\n"
        "    loss_ref = ((ref_model(X) - Y) ** 2).mean()\n"
        "    loss_ref.backward()\n"
        "    ref_opt.step()\n"
        "    # Our step.\n"
        "    our_opt.zero_grad()\n"
        "    loss_our = ((our_model(X) - Y) ** 2).mean()\n"
        "    loss_our.backward()\n"
        "    our_opt.step()\n"
        "    # Parameters must match to within float tolerance.\n"
        "    assert t.allclose(ref_model.weight.detach(), our_model.weight.detach(), atol=1e-5), (\n"
        "        f'step {step_i}: weight mismatch vs torch.optim.SGD'\n"
        "    )\n"
        "    assert t.allclose(ref_model.bias.detach(), our_model.bias.detach(), atol=1e-5), (\n"
        "        f'step {step_i}: bias mismatch'\n"
        "    )"
    ),
    "solution_body": (
        "class Ex2MomentumSGD:\n"
        "    def __init__(self, params, lr, momentum=0.9):\n"
        "        self.params = list(params)\n"
        "        self.lr = lr\n"
        "        self.momentum = momentum\n"
        "        self.bufs = [t.zeros_like(p) for p in self.params]\n"
        "\n"
        "    @t.inference_mode()\n"
        "    def step(self):\n"
        "        for p, b in zip(self.params, self.bufs):\n"
        "            if p.grad is None:\n"
        "                continue\n"
        "            b.copy_(self.momentum * b + p.grad)\n"
        "            p -= self.lr * b\n"
        "\n"
        "    def zero_grad(self):\n"
        "        for p in self.params:\n"
        "            p.grad = None"
    ),
    "solution_notes": (
        "**Why a buffer LIST parallel to params.** Each Parameter has a "
        "different shape, so the buffers can't be one big contiguous "
        "tensor (without flattening). The parallel-list pattern is what "
        "PyTorch's `torch.optim.SGD` does internally too — `state[p]` is "
        "a dict keyed by Parameter that stores the velocity buffer.\n\n"
        "**Why `b.copy_(...)` not `b = ...`.** Identical issue to "
        "EMA-first-moment ex1: rebinding the local loop variable doesn't "
        "update the list entry. The next step would read a stale zero "
        "buffer and behave like plain SGD with no momentum.\n\n"
        "**Match to torch.optim.SGD.** PyTorch's default formula is "
        "`v_new = momentum*v + g` (no `(1-mu)` factor — this is the "
        "classical-momentum form, not the EMA form). Our impl matches "
        "exactly, which is why the trajectories agree to 1e-5 over 10 "
        "steps. If you wanted the EMA-style "
        "`v_new = momentum*v + (1-momentum)*g` you'd need a different "
        "lr scale — that's Adam's first moment, not SGD."
    ),
}


# ============================================================ ATOM 7 / 8
# step-counter-increment — ex2: log_every interval gating
SPEC_STEP_COUNTER = {
    "atom_id": "step-counter-increment",
    "subtopic": "Trainer: step counter increment",
    "topic_folder": "prereqs_adam_trainer",
    "atom_recap_md": (
        "## `log_every` interval logging — quick refresher\n"
        "\n"
        "Logging every step is wasteful for long runs — wandb's free tier "
        "throttles around 50 log/s, and per-step disk I/O adds up. The "
        "canonical guard is an interval check `if self.step % log_every "
        "== 0`:\n"
        "```python\n"
        "self.optimizer.step()\n"
        "self.optimizer.zero_grad()\n"
        "self.step += 1                              # tick FIRST\n"
        "if self.step % log_every == 0:              # then gate\n"
        "    self.log({'step': self.step, 'loss': loss.item()})\n"
        "```\n"
        "Tick THEN gate so step 100 logs at step==100 (not 99). With "
        "`log_every=10` you get logs at steps 10, 20, 30, ..., 100, 110."
    ),
    "exercise_index": 2,
    "exercise_title": "log every N steps using modulo gate after step counter tick",
    "slug": "log-every-n-steps-modulo-gate-after-tick",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["log-every", "interval-logging", "modulo-gate", "step-counter"],
    "kcs": ["step-counter-increments-after-optimizer-step", "log-every-N-steps-modulo-gate"],
    "lo": (
        "Apply the `step % log_every == 0` interval-gate AFTER the step "
        "counter tick so logged step values are multiples of `log_every` "
        "(10, 20, 30...) rather than off-by-one (9, 19, 29...)."
    ),
    "prompt_body": (
        "Implement `ex2_train_with_log_every(losses, log_every)`. A "
        "training-loop-like driver that ticks a step counter every batch "
        "but only LOGS every `log_every` steps.\n\n"
        "1. Initialize `step = 0`, `log = []`.\n"
        "2. For each loss in `losses`:\n"
        "   - (Simulated update happens here — nothing to compute.)\n"
        "   - `step += 1` — tick AFTER the simulated update.\n"
        "   - If `step % log_every == 0`: `log.append((step, loss))`.\n"
        "3. Return `(step, log)`.\n"
        "\n"
        "Inputs:\n"
        "- `losses`: list of per-step floats.\n"
        "- `log_every`: int >= 1 — emit a log entry every this-many steps.\n"
        "\n"
        "Output:\n"
        "- `final_step`: int — equal to `len(losses)`.\n"
        "- `log`: list of `(step, loss)` tuples, with `step` taking values "
        "`log_every, 2*log_every, 3*log_every, ...`.\n\n"
        "**Critical:** the tick MUST happen BEFORE the modulo check, "
        "otherwise step 10 would log at step==9 (off-by-one)."
    ),
    "stub": (
        "def ex2_train_with_log_every(losses: list, log_every: int) -> tuple:\n"
        '    """Tick step every batch; log only when step % log_every == 0."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === log_every=10 with 100 losses → log at steps 10, 20, ..., 100 ===\n"
        "losses = [1.0 / (i + 1) for i in range(100)]\n"
        "final, log = ex2_train_with_log_every(losses, log_every=10)\n"
        "assert final == 100, f'final_step should be 100; got {final}'\n"
        "\n"
        "logged_steps = [s for s, _ in log]\n"
        "assert logged_steps == [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], (\n"
        "    f'expected step multiples of 10 up to 100; got {logged_steps}'\n"
        ")\n"
        "assert len(log) == 10, f'should log 10 times in 100-step run; got {len(log)}'\n"
        "\n"
        "# Loss values at the logged steps match losses[step-1].\n"
        "for step, loss in log:\n"
        "    expected_loss = losses[step - 1]\n"
        "    assert loss == expected_loss, (\n"
        "        f'log at step {step} has loss {loss}; expected losses[{step-1}]={expected_loss}'\n"
        "    )\n"
        "\n"
        "# === log_every=1 → log every step ===\n"
        "final2, log2 = ex2_train_with_log_every([0.5, 0.4, 0.3], log_every=1)\n"
        "assert final2 == 3\n"
        "assert [s for s, _ in log2] == [1, 2, 3], f'log_every=1 should log every step; got {log2}'\n"
        "\n"
        "# === log_every greater than n_losses → ZERO log entries ===\n"
        "final3, log3 = ex2_train_with_log_every([0.5, 0.4, 0.3], log_every=10)\n"
        "assert final3 == 3, 'step counter still ticks even if no log entries'\n"
        "assert log3 == [], f'log_every=10 with 3 steps should produce no logs; got {log3}'\n"
        "\n"
        "# === Edge: log_every exactly equals n_losses → exactly ONE log at end ===\n"
        "final4, log4 = ex2_train_with_log_every([0.1, 0.2, 0.3, 0.4, 0.5], log_every=5)\n"
        "assert final4 == 5\n"
        "assert log4 == [(5, 0.5)], f'log_every=5 with 5 steps should log exactly (5, 0.5); got {log4}'\n"
        "\n"
        "# === Off-by-one detector: 9-step run with log_every=3 logs at 3, 6, 9 NOT 2, 5, 8 ===\n"
        "final5, log5 = ex2_train_with_log_every([float(i) for i in range(9)], log_every=3)\n"
        "logged5 = [s for s, _ in log5]\n"
        "assert logged5 == [3, 6, 9], (\n"
        "    f'log_every=3 should log at multiples of 3 (tick BEFORE check); got {logged5}. '\n"
        "    f'If you got [2, 5, 8] you incremented step AFTER the modulo check.'\n"
        ")\n"
        "\n"
        "# === Empty losses → no logs, step=0 ===\n"
        "final6, log6 = ex2_train_with_log_every([], log_every=5)\n"
        "assert final6 == 0 and log6 == []"
    ),
    "solution_body": (
        "def ex2_train_with_log_every(losses, log_every):\n"
        "    step = 0\n"
        "    log = []\n"
        "    for loss in losses:\n"
        "        step += 1\n"
        "        if step % log_every == 0:\n"
        "            log.append((step, loss))\n"
        "    return step, log"
    ),
    "solution_notes": (
        "**Why tick BEFORE the modulo gate.** Logging at step 10 should "
        "reflect the state AFTER 10 updates have happened — same logic "
        "as ex1 in this folder. Tick first, then check `step % "
        "log_every`. If you reverse the order you log at step==9 instead "
        "of step==10 — the silent off-by-one that confuses 'why does my "
        "log file say step 9 when I asked for log_every=10?'\n\n"
        "**`log_every=1` is the no-throttle case.** Useful for short "
        "debug runs where you want every step. The modulo check still "
        "works (`step % 1 == 0` always True).\n\n"
        "**Variant: log on epoch boundary AND interval.** Real trainers "
        "often combine `if step % log_every == 0 OR end_of_epoch`. The "
        "interval catches mid-epoch detail; the epoch boundary catches "
        "the final post-validate state. Both gates live AFTER the tick."
    ),
}


# ============================================================ ATOM 8 / 8
# trainer-class-skeleton — ex2: callbacks list (post-epoch hook)
SPEC_TRAINER_SKELETON = {
    "atom_id": "trainer-class-skeleton",
    "subtopic": "Trainer: Trainer class skeleton",
    "topic_folder": "prereqs_adam_trainer",
    "atom_recap_md": (
        "## Trainer callbacks — quick refresher\n"
        "\n"
        "Once `fit / validate / _step` work, the next extension point is "
        "POST-EPOCH HOOKS. Lightning calls them callbacks; ARENA calls "
        "them \"on_epoch_end\" functions. The pattern: a list of "
        "callables that the Trainer invokes after each epoch, receiving "
        "the trainer itself so they can read `self.step`, "
        "`self.history`, `self.model`, etc.\n"
        "\n"
        "```python\n"
        "class Trainer:\n"
        "    def __init__(self, ...):\n"
        "        ...\n"
        "        self.callbacks = []          # list of fn(trainer) -> None\n"
        "\n"
        "    def fit(self, n_epochs):\n"
        "        for epoch in range(n_epochs):\n"
        "            self._train_epoch()\n"
        "            self.validate()\n"
        "            for cb in self.callbacks:\n"
        "                cb(self)             # hook point — runs AFTER validate\n"
        "```\n"
        "\n"
        "This is the right hook point because validate has already "
        "appended this epoch's val loss to `history`, so a callback can "
        "read it (early stopping, LR scheduling, checkpointing all work "
        "here)."
    ),
    "exercise_index": 2,
    "exercise_title": "Trainer with on_epoch_end callbacks list",
    "slug": "trainer-with-on-epoch-end-callbacks-list",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["callbacks", "on-epoch-end", "hooks", "trainer-extension"],
    "kcs": ["trainer-fit-loop-walks-epochs", "trainer-callbacks-fire-after-validate"],
    "lo": (
        "Apply the callback-list extension to the Trainer skeleton: a "
        "`self.callbacks` list of `fn(trainer)` callables that fire "
        "after each `validate()` call inside `fit`."
    ),
    "prompt_body": (
        "Implement `Ex2TrainerWithCallbacks`. Same shape as the minimal "
        "Trainer from ex1, plus a callbacks list.\n\n"
        "1. `__init__(self, model, optimizer, train_loader, val_loader, "
        "loss_fn)`:\n"
        "   - Store the five args; `self.step = 0`; "
        "`self.history = {'train_loss': [], 'val_loss': []}`.\n"
        "   - **NEW:** `self.callbacks = []` — list of `fn(trainer) -> "
        "None` callables.\n"
        "\n"
        "2. `_step(self, x, y)`: forward + loss (`self.loss_fn(self.model(x), "
        "y)`).\n"
        "\n"
        "3. `fit(self, n_epochs)`: per epoch:\n"
        "   - `self.model.train()`.\n"
        "   - iterate train_loader: `loss = self._step(x, y); "
        "loss.backward(); optimizer.step(); optimizer.zero_grad(); "
        "self.step += 1; history['train_loss'].append(loss.item())`.\n"
        "   - `self.validate()`.\n"
        "   - **NEW:** `for cb in self.callbacks: cb(self)` — runs AFTER "
        "validate so callbacks see this epoch's `history['val_loss']`.\n"
        "\n"
        "4. `validate(self)`: eval mode + inference_mode; "
        "weighted-by-batch-size val loss appended to "
        "`history['val_loss']`.\n"
        "\n"
        "The test registers TWO callbacks: an EpochCounter (just counts "
        "calls) and an EarlyStopRecorder (snapshots `history['val_loss'"
        "][-1]` after each epoch). The test verifies both fire exactly "
        "`n_epochs` times AND that the val-loss snapshot is the value "
        "freshly-appended by THIS epoch's validate (not stale)."
    ),
    "stub": (
        "class Ex2TrainerWithCallbacks:\n"
        "    \"\"\"Trainer + on_epoch_end callback list.\"\"\"\n"
        "\n"
        "    def __init__(self, model, optimizer, train_loader, val_loader, loss_fn):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    def _step(self, x, y):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    def fit(self, n_epochs: int):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    def validate(self):\n"
        "        raise NotImplementedError()"
    ),
    "test_body": (
        "from torch.utils.data import TensorDataset, DataLoader\n"
        "\n"
        "t.manual_seed(0)\n"
        "N = 64\n"
        "x_train = t.randn(N, 1)\n"
        "y_train = 2.0 * x_train + 1.0 + 0.05 * t.randn(N, 1)\n"
        "x_val = t.randn(16, 1)\n"
        "y_val = 2.0 * x_val + 1.0\n"
        "\n"
        "train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=8, shuffle=True)\n"
        "val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=8, shuffle=False)\n"
        "\n"
        "model = t.nn.Linear(1, 1)\n"
        "opt = t.optim.SGD(model.parameters(), lr=0.1)\n"
        "loss_fn = t.nn.MSELoss()\n"
        "\n"
        "trainer = Ex2TrainerWithCallbacks(model, opt, train_loader, val_loader, loss_fn)\n"
        "\n"
        "# Required attributes including the new callbacks list.\n"
        "assert hasattr(trainer, 'callbacks'), 'must have self.callbacks list'\n"
        "assert trainer.callbacks == [], f'callbacks must start empty; got {trainer.callbacks}'\n"
        "assert trainer.step == 0\n"
        "assert trainer.history == {'train_loss': [], 'val_loss': []}\n"
        "\n"
        "# === Register two callbacks ===\n"
        "epoch_count = {'n': 0}\n"
        "def epoch_counter(trainer_self):\n"
        "    epoch_count['n'] += 1\n"
        "\n"
        "val_snapshots = []\n"
        "def early_stop_recorder(trainer_self):\n"
        "    # Read the fresh val-loss appended by THIS epoch's validate.\n"
        "    val_snapshots.append(trainer_self.history['val_loss'][-1])\n"
        "\n"
        "trainer.callbacks.append(epoch_counter)\n"
        "trainer.callbacks.append(early_stop_recorder)\n"
        "\n"
        "# === Fit 4 epochs ===\n"
        "trainer.fit(n_epochs=4)\n"
        "\n"
        "# Each callback fired exactly n_epochs times.\n"
        "assert epoch_count['n'] == 4, f'epoch_counter should fire 4×; got {epoch_count[\"n\"]}'\n"
        "assert len(val_snapshots) == 4, f'early_stop_recorder should fire 4×; got {len(val_snapshots)}'\n"
        "\n"
        "# Snapshots must MATCH history['val_loss'] (callbacks ran AFTER validate).\n"
        "assert val_snapshots == trainer.history['val_loss'], (\n"
        "    f'callback snapshots must equal history[val_loss]; '\n"
        "    f'snap={val_snapshots} hist={trainer.history[\"val_loss\"]}'\n"
        ")\n"
        "\n"
        "# Validate that step counter and training loss history are right.\n"
        "expected_steps = 4 * len(train_loader)\n"
        "assert trainer.step == expected_steps\n"
        "assert len(trainer.history['train_loss']) == expected_steps\n"
        "assert len(trainer.history['val_loss']) == 4\n"
        "\n"
        "# === Loss decreased ===\n"
        "assert trainer.history['val_loss'][-1] < trainer.history['val_loss'][0], (\n"
        "    'val loss should decrease across 4 epochs'\n"
        ")\n"
        "\n"
        "# === No-callback path still works ===\n"
        "t.manual_seed(0)\n"
        "model2 = t.nn.Linear(1, 1)\n"
        "opt2 = t.optim.SGD(model2.parameters(), lr=0.1)\n"
        "trainer2 = Ex2TrainerWithCallbacks(model2, opt2, train_loader, val_loader, loss_fn)\n"
        "# Don't register any callbacks.\n"
        "trainer2.fit(n_epochs=2)\n"
        "assert len(trainer2.history['val_loss']) == 2, 'fit must work with empty callbacks list'\n"
        "\n"
        "# === Callback fires AFTER validate (sees the fresh val_loss, not stale) ===\n"
        "# Re-run with a callback that compares len(history[val_loss]) to the epoch number.\n"
        "t.manual_seed(0)\n"
        "model3 = t.nn.Linear(1, 1)\n"
        "opt3 = t.optim.SGD(model3.parameters(), lr=0.1)\n"
        "trainer3 = Ex2TrainerWithCallbacks(model3, opt3, train_loader, val_loader, loss_fn)\n"
        "observed_lens = []\n"
        "def observe_history_len(self):\n"
        "    observed_lens.append(len(self.history['val_loss']))\n"
        "trainer3.callbacks.append(observe_history_len)\n"
        "trainer3.fit(n_epochs=3)\n"
        "assert observed_lens == [1, 2, 3], (\n"
        "    f'callback should fire AFTER validate appends this epoch\\'s val_loss; '\n"
        "    f'expected [1, 2, 3], got {observed_lens}. '\n"
        "    f'If [0, 1, 2], your callback runs BEFORE validate.'\n"
        ")"
    ),
    "solution_body": (
        "class Ex2TrainerWithCallbacks:\n"
        "    def __init__(self, model, optimizer, train_loader, val_loader, loss_fn):\n"
        "        self.model = model\n"
        "        self.optimizer = optimizer\n"
        "        self.train_loader = train_loader\n"
        "        self.val_loader = val_loader\n"
        "        self.loss_fn = loss_fn\n"
        "        self.step = 0\n"
        "        self.history = {'train_loss': [], 'val_loss': []}\n"
        "        self.callbacks = []\n"
        "\n"
        "    def _step(self, x, y):\n"
        "        return self.loss_fn(self.model(x), y)\n"
        "\n"
        "    def fit(self, n_epochs):\n"
        "        for _epoch in range(n_epochs):\n"
        "            self.model.train()\n"
        "            for x, y in self.train_loader:\n"
        "                loss = self._step(x, y)\n"
        "                loss.backward()\n"
        "                self.optimizer.step()\n"
        "                self.optimizer.zero_grad()\n"
        "                self.step += 1\n"
        "                self.history['train_loss'].append(loss.item())\n"
        "            self.validate()\n"
        "            for cb in self.callbacks:\n"
        "                cb(self)\n"
        "\n"
        "    def validate(self):\n"
        "        self.model.eval()\n"
        "        total = 0.0\n"
        "        count = 0\n"
        "        with t.inference_mode():\n"
        "            for x, y in self.val_loader:\n"
        "                loss = self.loss_fn(self.model(x), y)\n"
        "                total += loss.item() * x.shape[0]\n"
        "                count += x.shape[0]\n"
        "        self.history['val_loss'].append(total / count)"
    ),
    "solution_notes": (
        "**Why callbacks run AFTER validate.** A callback that wants to "
        "trigger early stopping on val loss MUST see this epoch's val "
        "loss in `history['val_loss']`. Running before validate would "
        "give it stale data (last epoch's loss). The order "
        "train → validate → callbacks is the same one PyTorch Lightning "
        "uses (`on_validation_end` → `on_epoch_end`).\n\n"
        "**Why `cb(self)` and not bound methods.** Passing the trainer "
        "itself lets callbacks be plain functions — easiest to write and "
        "test. They can read anything on the trainer (history, model, "
        "step, optimizer LR groups) without inheritance. Lightning uses "
        "the same convention with its `Callback` class taking a "
        "`trainer` argument in every hook.\n\n"
        "**Common callbacks you'd register in real code.** "
        "`ModelCheckpoint(save_best=True)` reads "
        "`history['val_loss'][-1]` and saves model weights when it "
        "improves. `EarlyStopping(patience=5)` raises a sentinel to "
        "break fit early. `LRScheduler.step()` adjusts the optimizer LR "
        "based on the new val loss. All three only need the trainer "
        "reference and the post-validate moment."
    ),
}


# === Emit all 8 ===
ALL_SPECS = [
    SPEC_ARGMAX_ACCURACY,
    SPEC_BIAS_CORRECTION,
    SPEC_CROSS_ENTROPY,
    SPEC_EMA_FIRST_MOMENT,
    SPEC_EXAMPLES_SEEN,
    SPEC_OPTIMIZER_LOOP,
    SPEC_STEP_COUNTER,
    SPEC_TRAINER_SKELETON,
]


def _verify_all(specs):
    """Exec stub + solution + test in a fresh ns per spec; assert pass."""
    import torch as t
    import numpy as np
    from torch import Tensor
    import einops
    from einops import rearrange, reduce, repeat

    # Headless matplotlib for any spec that plots.
    import matplotlib
    matplotlib.use('Agg')

    passed = 0
    failed = []

    for spec in specs:
        ex_id = f"ex{spec['exercise_index']}"
        tag = f"{spec['atom_id']}/{ex_id}"
        ns = {
            "t": t,
            "np": np,
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
            # Stub may include NotImplementedError-raising bodies at module level for safety; tolerate.
            pass

        try:
            exec(spec["solution_body"], ns)
            exec(spec["test_body"], ns)
        except Exception as e:
            failed.append((tag, repr(e), traceback.format_exc()))
            print(f"  [verify] {tag}: FAIL — {e!r}")
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
    print(f"[deepening_h_batch9] Verifying {len(ALL_SPECS)} specs...")
    _verify_all(ALL_SPECS)

    print(f"\n[deepening_h_batch9] All verified — emitting notebooks.")
    for spec in ALL_SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")


if __name__ == "__main__":
    main()
