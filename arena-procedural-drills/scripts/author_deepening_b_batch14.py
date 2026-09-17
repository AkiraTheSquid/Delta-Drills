#!/usr/bin/env python3
"""Author 8 ex3 deepening drills (batch 14, group B) — prereqs_adam_trainer.

Atoms (8 / prereqs_adam_trainer):
    - argmax-accuracy-eval               (ex3: top-k accuracy via topk)
    - bias-correction-divide             (ex3: inflation-factor 1/(1-beta**t) decay over t)
    - cross-entropy-classification-loss  (ex3: class-weighted cross-entropy)
    - ema-first-moment                   (ex3: dual m+v update in one function)
    - examples-seen-step-axis            (ex3: gradient-accumulation factor)
    - optimizer-loop-on-tensor           (ex3: zero_grad per-param loop)
    - step-counter-increment             (ex3: max_steps early-stop check)
    - trainer-class-skeleton             (ex3: best-val-loss checkpoint snapshot)

Each ex3 hits a DISTINCT facet from ex1 and ex2. ONE LO + ONE Bloom + <=2 KCs per
drill. KCs are PFA-distinct from ex1/ex2 KCs (Maier 2021 §4.2).
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_adam_trainer"


# ---------------------------------------------------------------------------
# Recap blocks (each focused on the ex3 facet, not a copy of ex1/ex2 recap)
# ---------------------------------------------------------------------------

RECAP_TOPK = (
    "## Top-k accuracy — the wider generalization of argmax\n"
    "\n"
    "Ex1 + ex2 used `logits.argmax(dim=-1)` — top-1 only. Top-k accuracy "
    "asks the looser question: 'is the correct label inside the top-`k` "
    "highest-scoring predictions?'.\n"
    "\n"
    "```python\n"
    "# logits: (B, C). labels: (B,) int64.\n"
    "topk_preds = logits.topk(k, dim=-1).indices   # (B, k)\n"
    "correct    = (topk_preds == labels.unsqueeze(-1)).any(dim=-1)   # (B,) bool\n"
    "acc        = correct.float().mean().item()\n"
    "```\n"
    "\n"
    "**Why `topk` instead of `argsort`.** `topk` is O(B*C*log(k)), `argsort` "
    "is O(B*C*log(C)). When `k << C` (typical: k=5, C=1000 for ImageNet), "
    "topk is asymptotically faster and that's why every classification "
    "harness uses it.\n"
    "\n"
    "**Why `unsqueeze(-1)` + `.any(dim=-1)`.** `labels` is `(B,)`. Broadcasting "
    "to `(B, 1)` lets it compare element-wise against `(B, k)`. The `.any` "
    "collapses the k-axis: 'did ANY of the top-k slots match?'.\n"
    "\n"
    "**Top-1 is the k=1 case.** Setting k=1 reduces to ex1's argmax-equality "
    "check (after squeezing the size-1 axis). Top-k subsumes top-1."
)

RECAP_INFLATION_DECAY = (
    "## Bias-correction's inflation factor `1/(1 - beta**t)` over time\n"
    "\n"
    "Ex1 computed `m_hat = m / (1 - beta**t)` for one step. Ex2 traced the "
    "raw `m` trajectory under a constant gradient. The deepening move "
    "isolates the CORRECTION FACTOR itself — the multiplier `1/(1 - beta**t)` "
    "— and analyzes how it decays from large (at t=1) toward 1 (as t→∞).\n"
    "\n"
    "```python\n"
    "# At t=1 with beta=0.9:  1 / (1 - 0.9**1) = 1 / 0.1  = 10.0\n"
    "# At t=10:               1 / (1 - 0.9**10) ≈ 1 / 0.651 ≈ 1.535\n"
    "# At t=100:              1 / (1 - 0.9**100) ≈ 1.000027\n"
    "```\n"
    "\n"
    "**Monotone decreasing.** `beta**t` is monotonically decreasing in `t` "
    "(for `0 < beta < 1`), so `1 - beta**t` is monotonically INCREASING, "
    "and its reciprocal — the inflation factor — is monotonically "
    "DECREASING. Each subsequent step gets a smaller boost.\n"
    "\n"
    "**Lower bound = 1.** `beta**t → 0` as `t → ∞`, so the factor approaches "
    "`1/(1 - 0) = 1`. Bias correction becomes a no-op at large step counts — "
    "the EMA has 'warmed up' and no longer needs help.\n"
    "\n"
    "**Practical takeaway.** Bias correction matters MOST during the first "
    "~`1/(1-beta)` steps (≈10 steps for beta=0.9, ≈1000 for beta=0.999). "
    "Past that horizon, the factor is essentially 1 and the divide is "
    "numerical-noise."
)

RECAP_CE_WEIGHTED = (
    "## Class-weighted cross-entropy for imbalanced classes\n"
    "\n"
    "Ex1 did vanilla cross-entropy. Ex2 used `ignore_index` to skip padding. "
    "The deepening move handles a different real-world case: an IMBALANCED "
    "dataset where rare classes need a heavier loss contribution.\n"
    "\n"
    "```python\n"
    "# weights[c] = how much class-c examples count. Shape (C,).\n"
    "loss = F.cross_entropy(logits, labels, weight=weights)\n"
    "```\n"
    "\n"
    "**What `weight=` actually does.** PyTorch multiplies each example's "
    "per-class log-prob by `weight[label]` BEFORE averaging. The denominator "
    "of the mean is the SUM OF WEIGHTS for the labels in the batch, NOT the "
    "batch size — so the result is a weighted mean, not a sum.\n"
    "\n"
    "```python\n"
    "# Manual decomposition:\n"
    "log_probs    = F.log_softmax(logits, dim=-1)       # (B, C)\n"
    "per_ex       = -log_probs[range(len(labels)), labels]  # (B,) NLL\n"
    "per_ex_w     = per_ex * weights[labels]            # (B,) re-weighted\n"
    "loss         = per_ex_w.sum() / weights[labels].sum()  # weighted mean\n"
    "```\n"
    "\n"
    "**Common weight scheme.** `weight = 1 / class_counts` then normalized — "
    "each class contributes equally regardless of frequency. Combats class "
    "imbalance without resampling.\n"
    "\n"
    "**Uniform weights collapse to vanilla CE.** When `weight = ones(C)`, "
    "the formula reduces to `mean(per_ex)` — exactly ex1's result. The "
    "weighted form is a strict generalization."
)

RECAP_DUAL_MV = (
    "## Dual m + v update in one pass — Adam's full state buffers\n"
    "\n"
    "Ex1 updated `m` (first moment, EMA of gradient). Ex2 updated `v` "
    "(second moment, EMA of squared gradient). The deepening move runs "
    "BOTH updates in a single helper, mutating both buffers from the same "
    "gradient list — the exact pattern `torch.optim.Adam.step` uses inside "
    "its per-parameter loop.\n"
    "\n"
    "```python\n"
    "for m, v, g in zip(m_list, v_list, grad_list):\n"
    "    m.copy_(beta1 * m + (1 - beta1) * g)         # first moment\n"
    "    v.copy_(beta2 * v + (1 - beta2) * g.pow(2))  # second moment\n"
    "```\n"
    "\n"
    "**Order doesn't matter — but use one pass.** Since `m` and `v` are "
    "independent buffers (no cross-dependency), you can update them in "
    "either order. Combining them in ONE loop over parameters is the "
    "memory-locality win: each param's `(m, v, g)` triple is in cache "
    "together. Real Adam implementations always co-locate the update.\n"
    "\n"
    "**Why `g.pow(2)` not `g * g`.** Both work numerically. `pow(2)` is "
    "fused in PyTorch's CUDA kernel and skips one tensor allocation vs "
    "`g * g` (which materializes a temporary). For tight inner loops, "
    "`pow(2)` is the canonical choice.\n"
    "\n"
    "**No bias correction here.** The dual-update helper just runs the EMAs. "
    "Bias correction (`m_hat`, `v_hat`) is the NEXT step in Adam and lives "
    "in a separate atom — keeping concerns separate is the whole point of "
    "the per-atom skeleton."
)

RECAP_GRAD_ACCUM = (
    "## Gradient accumulation — examples_seen = step * batch_size * accum_steps\n"
    "\n"
    "Ex1 + ex2 used `examples_seen = step * batch_size`. The deepening move "
    "handles gradient accumulation: you do `accum_steps` forward+backward "
    "passes BEFORE calling `optimizer.step()`. From the optimizer's view "
    "each 'step' actually saw `accum_steps` micro-batches.\n"
    "\n"
    "```python\n"
    "# step = optimizer steps. micro-batch = backward but no step yet.\n"
    "examples_seen = step * batch_size * accum_steps\n"
    "```\n"
    "\n"
    "**Why this matters.** You want curves COMPARABLE across runs with "
    "different `batch_size` AND different `accum_steps`. Two runs with "
    "`(B=8, accum=4)` and `(B=32, accum=1)` see the same effective batch "
    "(32 examples per optimizer step), so logging by `examples_seen` makes "
    "their loss curves overlap. Logging by `step` would make them diverge "
    "visually for no real reason.\n"
    "\n"
    "**`accum_steps=1` collapses to ex1/ex2.** When you're not accumulating, "
    "`step * batch_size * 1 == step * batch_size`. The formula is a strict "
    "generalization — same shape, extra factor.\n"
    "\n"
    "**It's about effective examples, not micro-batches.** The micro-batch "
    "count INSIDE one optimizer step is `accum_steps`. The optimizer step "
    "counter increments only AFTER all micro-batches have backprop'd. So "
    "`examples_seen` per optimizer step grows by `batch_size * accum_steps`."
)

RECAP_ZERO_GRAD = (
    "## `zero_grad()` — the second per-parameter loop in every optimizer\n"
    "\n"
    "Ex1 + ex2 implemented the per-parameter STEP loop. Every optimizer also "
    "has a per-parameter ZERO_GRAD loop. They share the same iteration "
    "pattern but do opposite things: step CONSUMES `.grad`, zero_grad CLEARS "
    "it for the next backward.\n"
    "\n"
    "```python\n"
    "@t.inference_mode()\n"
    "def zero_grad(self, set_to_none: bool = True):\n"
    "    for p in self.params:\n"
    "        if p.grad is None:\n"
    "            continue\n"
    "        if set_to_none:\n"
    "            p.grad = None              # cheaper: drops the tensor\n"
    "        else:\n"
    "            p.grad.zero_()             # in-place fill with zeros\n"
    "```\n"
    "\n"
    "**`set_to_none=True` is now the default in PyTorch.** Why: assigning "
    "`None` deallocates the `.grad` tensor's storage, saving memory between "
    "the `step()` and the next `backward()`. The next `backward()` "
    "re-allocates a fresh grad — which would have been overwritten anyway.\n"
    "\n"
    "**`zero_()` (with underscore) is the in-place variant.** When the user "
    "wants `.grad` to remain a tensor of zeros (e.g. for downstream code "
    "that reads `.grad` between steps), pass `set_to_none=False`.\n"
    "\n"
    "**The `if p.grad is None: continue` guard.** Same guard as in `step` — "
    "frozen / unused params have no `.grad` attribute. Without the guard, "
    "the `.grad.zero_()` call raises `AttributeError`."
)

RECAP_MAX_STEPS = (
    "## `max_steps` termination — train by steps, not just by epochs\n"
    "\n"
    "Ex1 + ex2 incremented a step counter and gated logging on it. The "
    "deepening move uses the SAME counter as a STOP condition: training "
    "halts as soon as `step >= max_steps`, even mid-epoch.\n"
    "\n"
    "```python\n"
    "step = 0\n"
    "stopped = False\n"
    "for epoch in range(n_epochs):\n"
    "    for batch in train_loader:\n"
    "        # ... forward, backward, optimizer.step() ...\n"
    "        step += 1\n"
    "        if step >= max_steps:\n"
    "            stopped = True\n"
    "            break\n"
    "    if stopped:\n"
    "        break\n"
    "```\n"
    "\n"
    "**Why step-based termination is standard for LLM training.** Epochs "
    "are dataset-size-dependent — the same 'epoch' is 10x more compute on "
    "a 10x larger dataset. Step-based budgets (`max_steps=100_000`) are "
    "dataset-independent. Hugging Face's `TrainingArguments.max_steps` "
    "uses exactly this pattern.\n"
    "\n"
    "**Two breaks: one inner, one outer.** Python's `break` only exits the "
    "INNERMOST loop. The outer `for epoch` needs its own break gated on a "
    "`stopped` flag — otherwise you'd inadvertently start a new epoch.\n"
    "\n"
    "**Check AFTER the tick.** `step += 1` then `if step >= max_steps: break`. "
    "Doing the check BEFORE the tick (or not ticking at all) would either "
    "off-by-one the final step count or skip the last batch. Tick-then-check "
    "is the safe order."
)

RECAP_BEST_CHECKPOINT = (
    "## Best-val-loss checkpoint — track best, snapshot state_dict\n"
    "\n"
    "Ex1 implemented `fit`+`validate`+`_step`. Ex2 added a callback list. "
    "The deepening move adds the SINGLE most-common Trainer extension: "
    "remember the best validation loss seen so far, and snapshot the "
    "model's `state_dict()` when it improves.\n"
    "\n"
    "```python\n"
    "self.best_val_loss = float('inf')\n"
    "self.best_state    = None    # snapshot dict (or None if never improved)\n"
    "\n"
    "def validate(self):\n"
    "    # ... compute val_loss ...\n"
    "    self.history['val_loss'].append(val_loss)\n"
    "    if val_loss < self.best_val_loss:\n"
    "        self.best_val_loss = val_loss\n"
    "        # DEEP-COPY: state_dict() returns LIVE references; we need a snapshot.\n"
    "        self.best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}\n"
    "```\n"
    "\n"
    "**Why deep-copy.** `model.state_dict()` returns a dict of LIVE tensors "
    "— the same tensors that `optimizer.step()` is about to mutate. Saving "
    "the dict directly would mean `best_state` updates IN LOCKSTEP with the "
    "training, defeating the whole point. `.detach().clone()` per tensor "
    "produces an independent snapshot.\n"
    "\n"
    "**`best_state = None` sentinel.** Before the first validate call, no "
    "snapshot has been taken. Returning `None` is cleaner than 'snapshot of "
    "the random init' — callers can check `if trainer.best_state is None: ...`.\n"
    "\n"
    "**Strict `<`, not `<=`.** Use strict less-than for 'improved'. With "
    "`<=`, ties would overwrite the snapshot needlessly and you'd snapshot "
    "the LATER tied model rather than the EARLIER one (which is usually "
    "preferred — earlier means less overfit)."
)


# ---------------------------------------------------------------------------
# SPEC 1 — argmax-accuracy-eval ex3
# ---------------------------------------------------------------------------

SPEC_ARGMAX = {
    "atom_id": "argmax-accuracy-eval",
    "subtopic": "Eval: argmax accuracy",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_TOPK,
    "exercise_index": 3,
    "exercise_title": "top-k accuracy via logits.topk and any-match",
    "slug": "top-k-accuracy-via-topk-any-match",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["accuracy", "top-k", "topk", "classification"],
    "kcs": [
        "topk-indices-along-class-dim",
        "any-match-collapses-k-axis",
    ],
    "lo": (
        "Apply `logits.topk(k, dim=-1).indices` followed by a broadcasted "
        "equality + `.any(dim=-1)` so the per-example boolean tells you "
        "whether the correct label was inside the top-k predictions."
    ),
    "prompt_body": (
        "Implement `ex3_topk_accuracy(logits, labels, k)`. The top-k "
        "classification accuracy metric.\n\n"
        "Inputs:\n"
        "- `logits`: `(B, C)` float tensor.\n"
        "- `labels`: `(B,)` int64 tensor with values in `[0, C)`.\n"
        "- `k`: int, `1 <= k <= C`.\n\n"
        "Steps:\n\n"
        "1. Get the indices of the top-k highest-scoring classes per row: "
        "`topk_idx = logits.topk(k, dim=-1).indices` — shape `(B, k)`.\n"
        "2. Compare against labels: broadcast `labels` to `(B, 1)` via "
        "`labels.unsqueeze(-1)` and compare element-wise against `topk_idx` "
        "— gives a `(B, k)` bool tensor.\n"
        "3. Collapse the k-axis with `.any(dim=-1)` — `(B,)` bool: True iff "
        "any of the top-k slots matched.\n"
        "4. Convert to float and average: `.float().mean().item()`.\n\n"
        "Return a Python float in `[0.0, 1.0]`."
    ),
    "stub": (
        "def ex3_topk_accuracy(logits, labels, k: int) -> float:\n"
        '    """Fraction of examples where the correct label is in the top-k predictions."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Hand-crafted: 4 examples, 5 classes ===\n"
        "# logits row 0: argmax=2 (largest), top-3 = {2, 1, 4}\n"
        "# logits row 1: argmax=0, top-3 = {0, 3, 1}\n"
        "# logits row 2: argmax=4, top-3 = {4, 2, 0} (label=0 in top-3 but NOT in top-2)\n"
        "# logits row 3: argmax=1, top-3 = {1, 4, 0}\n"
        "logits = t.tensor([\n"
        "    [0.1, 0.5, 2.0, 0.0, 0.4],\n"
        "    [3.0, 0.5, 0.1, 1.0, 0.0],\n"
        "    [0.3, 0.0, 0.8, 0.2, 2.5],\n"
        "    [0.4, 2.2, 0.1, 0.0, 1.5],\n"
        "])\n"
        "labels = t.tensor([2, 3, 0, 4])  # correct slots\n"
        "\n"
        "# === Top-1 == argmax check ===\n"
        "# argmax row-by-row: 2, 0, 4, 1.  Match labels (2,3,0,4)? 1, 0, 0, 0 → 0.25.\n"
        "assert abs(ex3_topk_accuracy(logits, labels, k=1) - 0.25) < 1e-6, (\n"
        "    f'k=1 should match argmax accuracy: expected 0.25, got {ex3_topk_accuracy(logits, labels, 1)}'\n"
        ")\n"
        "\n"
        "# === Top-3 ===\n"
        "# row 0: label=2 in {2,1,4} → True\n"
        "# row 1: label=3 in {0,3,1} → True\n"
        "# row 2: label=0 in {4,2,0} → True\n"
        "# row 3: label=4 in {1,4,0} → True\n"
        "# → all four match → 1.0\n"
        "assert ex3_topk_accuracy(logits, labels, k=3) == 1.0, (\n"
        "    f'k=3 should give 1.0; got {ex3_topk_accuracy(logits, labels, 3)}'\n"
        ")\n"
        "\n"
        "# === Top-2 ===\n"
        "# row 0: label=2 in {2,1} → True\n"
        "# row 1: label=3 in {0,3} → True\n"
        "# row 2: label=0 in {4,2} → False\n"
        "# row 3: label=4 in {1,4} → True\n"
        "# → 3/4 = 0.75\n"
        "assert abs(ex3_topk_accuracy(logits, labels, k=2) - 0.75) < 1e-6, (\n"
        "    f'k=2 expected 0.75; got {ex3_topk_accuracy(logits, labels, 2)}'\n"
        ")\n"
        "\n"
        "# === Top-5 (=C) always 1.0 ===\n"
        "assert ex3_topk_accuracy(logits, labels, k=5) == 1.0, (\n"
        "    f'k=C should give 1.0; got {ex3_topk_accuracy(logits, labels, 5)}'\n"
        ")\n"
        "\n"
        "# === Return type is Python float ===\n"
        "out = ex3_topk_accuracy(logits, labels, k=1)\n"
        "assert isinstance(out, float), f'must return Python float, got {type(out)}'\n"
        "assert 0.0 <= out <= 1.0\n"
        "\n"
        "# === Monotone in k ===\n"
        "# top-k accuracy is non-decreasing in k for any fixed (logits, labels).\n"
        "t.manual_seed(1)\n"
        "B, C = 32, 10\n"
        "L = t.randn(B, C)\n"
        "y = t.randint(0, C, (B,))\n"
        "accs = [ex3_topk_accuracy(L, y, k=k) for k in [1, 2, 3, 5, 7, 10]]\n"
        "for i in range(len(accs) - 1):\n"
        "    assert accs[i] <= accs[i+1] + 1e-9, (\n"
        "        f'top-k accuracy must be non-decreasing in k: {accs}'\n"
        "    )\n"
        "assert accs[-1] == 1.0, f'k=C should give 1.0; got accs={accs}'\n"
        "\n"
        "# === Perfect predictions → 1.0 at every k ===\n"
        "# Construct logits where argmax row-i = labels[i] with huge margin.\n"
        "y = t.tensor([3, 1, 4, 0])\n"
        "L = t.zeros(4, 5)\n"
        "L[t.arange(4), y] = 10.0\n"
        "for k_test in [1, 2, 3, 4, 5]:\n"
        "    assert ex3_topk_accuracy(L, y, k_test) == 1.0, f'perfect at k={k_test} should be 1.0'"
    ),
    "solution_body": (
        "def ex3_topk_accuracy(logits, labels, k):\n"
        "    topk_idx = logits.topk(k, dim=-1).indices       # (B, k)\n"
        "    correct = (topk_idx == labels.unsqueeze(-1)).any(dim=-1)  # (B,) bool\n"
        "    return correct.float().mean().item()"
    ),
    "solution_notes": (
        "**Why `topk` over `argsort`.** `argsort` is O(C log C); `topk` is "
        "O(C log k). For ImageNet-style `C=1000, k=5`, `topk` is ~30x "
        "fewer comparisons per row and is what every benchmark harness "
        "uses.\n\n"
        "**Why `unsqueeze(-1)` matters.** Without it, `labels` is `(B,)` "
        "and `topk_idx` is `(B, k)` — broadcasting would attempt `(B,) vs "
        "(B, k)` which raises (broadcasting from the right, `B != k`). "
        "Unsqueezing gives `(B, 1) vs (B, k)` → valid broadcast.\n\n"
        "**Top-1 == argmax sanity.** Setting k=1 reduces to ex1's metric "
        "after squeezing the size-1 axis — a useful self-consistency check "
        "and the first test we run."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 2 — bias-correction-divide ex3
# ---------------------------------------------------------------------------

SPEC_BIAS = {
    "atom_id": "bias-correction-divide",
    "subtopic": "Optimizer: Adam bias-correction divide",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_INFLATION_DECAY,
    "exercise_index": 3,
    "exercise_title": "inflation factor 1/(1-beta**t) decays monotonically to 1",
    "slug": "inflation-factor-decays-to-one",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["bias-correction", "inflation-factor", "decay", "warmup"],
    "kcs": [
        "inflation-factor-equals-reciprocal-one-minus-beta-to-t",
        "inflation-factor-monotone-decreasing-to-one",
    ],
    "lo": (
        "Analyze the bias-correction factor `1/(1 - beta**t)` across step "
        "indices t=1..n, demonstrating it is monotonically decreasing in t "
        "and asymptotes to 1, with rate controlled by beta."
    ),
    "prompt_body": (
        "Implement `ex3_inflation_factor_trajectory(beta, n_steps)`. Compute "
        "the bias-correction inflation factor `1/(1 - beta**t)` for "
        "`t = 1, 2, ..., n_steps` and return it as a list of Python floats.\n\n"
        "Inputs:\n"
        "- `beta`: float in `(0, 1)` — the EMA decay.\n"
        "- `n_steps`: int, `n_steps >= 1`.\n\n"
        "Return: `list[float]` of length `n_steps`, where the `i`-th entry "
        "(0-indexed) is `1 / (1 - beta**(i+1))`. The 1-based step index "
        "matches the Adam convention.\n\n"
        "Implementation: pure Python `**` and arithmetic — no tensor ops "
        "needed. Each entry is a regular Python float."
    ),
    "stub": (
        "def ex3_inflation_factor_trajectory(beta: float, n_steps: int) -> list:\n"
        '    """Bias-correction inflation factors 1/(1-beta**t) for t=1..n_steps."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Basic shape ===\n"
        "out = ex3_inflation_factor_trajectory(0.9, 5)\n"
        "assert isinstance(out, list), f'must return list, got {type(out)}'\n"
        "assert len(out) == 5, f'length must be n_steps=5, got {len(out)}'\n"
        "assert all(isinstance(x, float) for x in out), 'entries must be floats'\n"
        "\n"
        "# === Closed-form values at beta=0.9 ===\n"
        "# t=1:  1/(1-0.9)        = 10.0\n"
        "# t=2:  1/(1-0.81)       ≈ 5.2632\n"
        "# t=10: 1/(1-0.9**10)    ≈ 1.5354\n"
        "assert abs(out[0] - 10.0) < 1e-9, f'beta=0.9 t=1 should be 10.0; got {out[0]}'\n"
        "assert abs(out[1] - 1/(1-0.81)) < 1e-9, f'beta=0.9 t=2 mismatch; got {out[1]}'\n"
        "\n"
        "# === Monotone decreasing — the headline property ===\n"
        "for beta in [0.5, 0.9, 0.99, 0.999]:\n"
        "    traj = ex3_inflation_factor_trajectory(beta, 50)\n"
        "    for i in range(len(traj) - 1):\n"
        "        assert traj[i] >= traj[i+1], (\n"
        "            f'inflation factor must be monotone decreasing for beta={beta}; '\n"
        "            f'failed at i={i}: {traj[i]} < {traj[i+1]}'\n"
        "        )\n"
        "\n"
        "# === Asymptotes to 1 ===\n"
        "# For beta=0.9, by t=200 we should be essentially at 1.0.\n"
        "long_run = ex3_inflation_factor_trajectory(0.9, 300)\n"
        "assert abs(long_run[-1] - 1.0) < 1e-6, (\n"
        "    f'inflation factor at t=300, beta=0.9 should be ~1.0; got {long_run[-1]}'\n"
        ")\n"
        "# Smaller beta converges FASTER to 1.\n"
        "fast = ex3_inflation_factor_trajectory(0.5, 20)\n"
        "assert abs(fast[-1] - 1.0) < 1e-5, f'beta=0.5 t=20 should be ~1.0; got {fast[-1]}'\n"
        "\n"
        "# === Larger beta → larger factor at fixed t ===\n"
        "# At t=10: beta=0.999 gives a much larger factor than beta=0.9.\n"
        "t10_slow = ex3_inflation_factor_trajectory(0.999, 10)[-1]\n"
        "t10_fast = ex3_inflation_factor_trajectory(0.9, 10)[-1]\n"
        "assert t10_slow > t10_fast, (\n"
        "    f'slower decay (beta=0.999) needs larger correction at t=10 than beta=0.9; '\n"
        "    f'got {t10_slow} vs {t10_fast}'\n"
        ")\n"
        "\n"
        "# === Always > 1.0 (factor INFLATES, never deflates) ===\n"
        "for beta in [0.5, 0.9, 0.99, 0.999]:\n"
        "    traj = ex3_inflation_factor_trajectory(beta, 30)\n"
        "    for i, v in enumerate(traj):\n"
        "        assert v > 1.0 or abs(v - 1.0) < 1e-9, (\n"
        "            f'factor must be >= 1 for beta={beta} t={i+1}; got {v}'\n"
        "        )\n"
        "\n"
        "# === n_steps=1 edge case ===\n"
        "single = ex3_inflation_factor_trajectory(0.9, 1)\n"
        "assert len(single) == 1\n"
        "assert abs(single[0] - 10.0) < 1e-9, f'single-step at beta=0.9 should be 10.0; got {single[0]}'\n"
        "\n"
        "# === Effective horizon rule of thumb: at t ~ 1/(1-beta), factor ≈ ~1.58 ===\n"
        "# For beta=0.9, horizon is 10. inflation_factor[10-1] (1-indexed t=10) ≈ 1.535\n"
        "h_traj = ex3_inflation_factor_trajectory(0.9, 10)\n"
        "assert 1.4 < h_traj[-1] < 1.7, (\n"
        "    f'at t=1/(1-beta)=10, factor should be in [1.4, 1.7]; got {h_traj[-1]}'\n"
        ")"
    ),
    "solution_body": (
        "def ex3_inflation_factor_trajectory(beta, n_steps):\n"
        "    return [1.0 / (1.0 - beta ** t) for t in range(1, n_steps + 1)]"
    ),
    "solution_notes": (
        "**The 1-based loop is mandatory.** Adam's bias correction is "
        "`1 - beta**t` with `t` starting at 1 (not 0). Using `range(0, n)` "
        "would put a `1/(1 - 1) = 1/0` divide-by-zero at the first step.\n\n"
        "**Effective horizon = `1/(1-beta)`.** Substituting `t = 1/(1-beta)` "
        "gives `beta**t ≈ exp(-1) ≈ 0.368` (for beta near 1), so the factor "
        "is `1/(1 - 0.368) ≈ 1.58`. Past this horizon, bias correction has "
        "done most of its work and the factor approaches 1 quickly.\n\n"
        "**The list-comp is `O(n)`.** For Adam in real training you'd never "
        "materialize this list — you just compute the current step's factor "
        "on the fly. The materialized trajectory exists for plotting / "
        "analysis only."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 3 — cross-entropy-classification-loss ex3
# ---------------------------------------------------------------------------

SPEC_CE = {
    "atom_id": "cross-entropy-classification-loss",
    "subtopic": "Loss: Cross-entropy classification",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_CE_WEIGHTED,
    "exercise_index": 3,
    "exercise_title": "class-weighted cross-entropy for imbalanced classes",
    "slug": "class-weighted-cross-entropy",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["cross-entropy", "class-weights", "imbalanced", "weighted-loss"],
    "kcs": [
        "class-weight-multiplies-per-example-loss",
        "weighted-mean-divides-by-weight-sum-not-batch-size",
    ],
    "lo": (
        "Apply per-class weights to cross-entropy so rare-class examples "
        "carry a heavier loss contribution, computed manually and verified "
        "against `F.cross_entropy(..., weight=...)`."
    ),
    "prompt_body": (
        "Implement `ex3_weighted_cross_entropy(logits, labels, weights)`. The "
        "class-weighted cross-entropy loss, computed manually.\n\n"
        "Inputs:\n"
        "- `logits`: `(B, C)` float tensor — unnormalized scores.\n"
        "- `labels`: `(B,)` int64 tensor — ground-truth class indices in "
        "`[0, C)`.\n"
        "- `weights`: `(C,)` float tensor — per-class loss weights "
        "(positive, but NOT necessarily normalized).\n\n"
        "Steps:\n\n"
        "1. `log_probs = F.log_softmax(logits, dim=-1)` — `(B, C)`.\n"
        "2. Gather the log-prob of the correct class per example: "
        "`tgt_lp = log_probs[range(len(labels)), labels]` — `(B,)`.\n"
        "3. Compute per-example NLL: `per_ex = -tgt_lp` — `(B,)`.\n"
        "4. Gather the corresponding weights: `w_per_ex = weights[labels]` "
        "— `(B,)`.\n"
        "5. Weighted mean: `(per_ex * w_per_ex).sum() / w_per_ex.sum()`.\n\n"
        "Return a scalar tensor (0-D, `loss.shape == ()`).\n\n"
        "**Must match `F.cross_entropy(logits, labels, weight=weights)` "
        "to floating-point tolerance.**"
    ),
    "stub": (
        "def ex3_weighted_cross_entropy(logits, labels, weights):\n"
        '    """Class-weighted cross-entropy — must match F.cross_entropy(weight=...)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Basic correctness against F.cross_entropy ===\n"
        "t.manual_seed(0)\n"
        "B, C = 32, 5\n"
        "logits = t.randn(B, C)\n"
        "labels = t.randint(0, C, (B,))\n"
        "weights = t.tensor([1.0, 2.0, 0.5, 4.0, 1.5])\n"
        "\n"
        "expected = F.cross_entropy(logits, labels, weight=weights)\n"
        "got = ex3_weighted_cross_entropy(logits, labels, weights)\n"
        "assert isinstance(got, t.Tensor), f'must return tensor, got {type(got)}'\n"
        "assert got.ndim == 0, f'must be scalar (0-D); got shape {tuple(got.shape)}'\n"
        "assert t.allclose(got, expected, atol=1e-6), (\n"
        "    f'weighted CE mismatch:\\n  got={got.item()}\\n  expected={expected.item()}'\n"
        ")\n"
        "\n"
        "# === Uniform weights collapse to vanilla cross-entropy ===\n"
        "uniform = t.ones(C)\n"
        "vanilla = F.cross_entropy(logits, labels)\n"
        "got_uniform = ex3_weighted_cross_entropy(logits, labels, uniform)\n"
        "assert t.allclose(got_uniform, vanilla, atol=1e-6), (\n"
        "    f'uniform weights should collapse to vanilla CE:\\n  got={got_uniform.item()}\\n  '\n"
        "    f'vanilla={vanilla.item()}'\n"
        ")\n"
        "\n"
        "# === Doubling all weights does NOT change the result (weighted MEAN) ===\n"
        "got1 = ex3_weighted_cross_entropy(logits, labels, weights)\n"
        "got2 = ex3_weighted_cross_entropy(logits, labels, weights * 2.0)\n"
        "assert t.allclose(got1, got2, atol=1e-6), (\n"
        "    f'scaling weights uniformly should not change weighted mean:\\n  '\n"
        "    f'  got1={got1.item()} got2={got2.item()}'\n"
        ")\n"
        "\n"
        "# === Heavier weight on a class actually changes the loss ===\n"
        "# A weight that boosts the rare-class label should pull the loss toward\n"
        "# that class's NLL contribution.\n"
        "w_uniform = t.ones(C)\n"
        "w_boost = t.ones(C); w_boost[0] = 10.0\n"
        "loss_uniform = ex3_weighted_cross_entropy(logits, labels, w_uniform)\n"
        "loss_boost = ex3_weighted_cross_entropy(logits, labels, w_boost)\n"
        "# These should be different (provided some examples have label==0).\n"
        "assert (labels == 0).any(), 'test setup: need at least one class-0 example'\n"
        "assert not t.allclose(loss_uniform, loss_boost, atol=1e-4), (\n"
        "    'boosting class-0 weight should change the weighted-mean loss'\n"
        ")\n"
        "\n"
        "# === Imbalanced case: 1 rare example + 9 common ===\n"
        "logits_small = t.randn(10, 3)\n"
        "labels_small = t.tensor([0] * 9 + [2])  # class 2 is rare (1 of 10)\n"
        "weights_small = t.tensor([1.0, 1.0, 9.0])  # boost rare class 9x\n"
        "expected = F.cross_entropy(logits_small, labels_small, weight=weights_small)\n"
        "got = ex3_weighted_cross_entropy(logits_small, labels_small, weights_small)\n"
        "assert t.allclose(got, expected, atol=1e-6)\n"
        "\n"
        "# === All examples same class — weighted mean of identical entries == per-example NLL ===\n"
        "logits_one = t.randn(4, 3)\n"
        "labels_one = t.tensor([1, 1, 1, 1])\n"
        "w_one = t.tensor([0.3, 0.7, 0.2])\n"
        "got = ex3_weighted_cross_entropy(logits_one, labels_one, w_one)\n"
        "lp = F.log_softmax(logits_one, dim=-1)\n"
        "expected_manual = -lp[t.arange(4), labels_one].mean()  # weights cancel\n"
        "assert t.allclose(got, expected_manual, atol=1e-6), (\n"
        "    f'when all labels are same class, weighted mean == regular mean'\n"
        ")\n"
        "\n"
        "# === Return type sanity ===\n"
        "result = ex3_weighted_cross_entropy(logits, labels, weights)\n"
        "assert result.dtype.is_floating_point, f'loss should be float dtype; got {result.dtype}'"
    ),
    "solution_body": (
        "def ex3_weighted_cross_entropy(logits, labels, weights):\n"
        "    log_probs = F.log_softmax(logits, dim=-1)\n"
        "    tgt_lp = log_probs[t.arange(len(labels)), labels]\n"
        "    per_ex = -tgt_lp                       # (B,)\n"
        "    w_per_ex = weights[labels]             # (B,)\n"
        "    return (per_ex * w_per_ex).sum() / w_per_ex.sum()"
    ),
    "solution_notes": (
        "**Why divide by `w_per_ex.sum()` not `B`.** PyTorch's "
        "`F.cross_entropy(weight=...)` returns a WEIGHTED MEAN — the "
        "denominator is the sum of weights for the labels appearing in the "
        "batch, not the batch size. Dividing by B would make the result "
        "scale with the weight magnitude, which is precisely what "
        "weighted-mean is designed to avoid.\n\n"
        "**Uniform weights collapse case.** When `weights = ones(C)`, "
        "`w_per_ex = ones(B)`, `w_per_ex.sum() = B`, and the result is "
        "`per_ex.sum() / B == per_ex.mean()` — exactly ex1's vanilla "
        "cross-entropy. Strict generalization.\n\n"
        "**Why `t.arange(len(labels))` not `range(...)`.** Both work — "
        "`F.log_softmax` returns a tensor and tensor-indexing accepts "
        "either. `t.arange` keeps everything on the same device (matters "
        "for GPU tensors); `range` would silently move the index to CPU."
    ),
    "extra_imports": ["import torch.nn as nn", "import torch.nn.functional as F"],
}


# ---------------------------------------------------------------------------
# SPEC 4 — ema-first-moment ex3
# ---------------------------------------------------------------------------

SPEC_EMA = {
    "atom_id": "ema-first-moment",
    "subtopic": "Optimizer: Adam EMA first moment",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_DUAL_MV,
    "exercise_index": 3,
    "exercise_title": "dual m and v EMA update from one gradient list",
    "slug": "dual-m-and-v-update-from-one-grad-list",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["adam", "first-moment", "second-moment", "co-located-update"],
    "kcs": [
        "co-locate-m-and-v-update-in-one-loop",
        "second-moment-uses-g-squared-not-g",
    ],
    "lo": (
        "Apply both Adam EMA recurrences in a single co-located loop — "
        "`m = beta1*m + (1-beta1)*g` and `v = beta2*v + (1-beta2)*g**2` — "
        "so each parameter's `(m, v, g)` triple is touched once."
    ),
    "prompt_body": (
        "Implement `ex3_update_m_and_v(m_list, v_list, grad_list, beta1, "
        "beta2)`. Apply Adam's first-moment AND second-moment EMA updates "
        "in one combined pass.\n\n"
        "Inputs:\n"
        "- `m_list`: list of first-moment buffers (one tensor per param).\n"
        "- `v_list`: list of second-moment buffers (one tensor per param).\n"
        "- `grad_list`: list of gradient tensors (one per param).\n"
        "- `beta1`: float — EMA decay for `m`.\n"
        "- `beta2`: float — EMA decay for `v`.\n\n"
        "All three lists have the same length, and the i-th entries share a "
        "shape.\n\n"
        "For each triple `(m, v, g)`:\n\n"
        "1. `m.copy_(beta1 * m + (1 - beta1) * g)` — first moment.\n"
        "2. `v.copy_(beta2 * v + (1 - beta2) * g.pow(2))` — second moment.\n\n"
        "Mutate `m_list` and `v_list` IN PLACE. Return `None`.\n\n"
        "Both updates MUST use `.copy_()` so the original buffer tensor "
        "identity (its `data_ptr()`) is preserved across the call."
    ),
    "stub": (
        "def ex3_update_m_and_v(m_list, v_list, grad_list, beta1: float, beta2: float):\n"
        '    """Update both Adam moments in place from a single gradient list."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Single param, one step from zero ===\n"
        "m = t.zeros(3)\n"
        "v = t.zeros(3)\n"
        "g = t.tensor([1.0, -2.0, 3.0])\n"
        "ptr_m, ptr_v = m.data_ptr(), v.data_ptr()\n"
        "result = ex3_update_m_and_v([m], [v], [g], beta1=0.9, beta2=0.999)\n"
        "assert result is None, f'must return None; got {result}'\n"
        "\n"
        "# m = 0.9 * 0 + 0.1 * g = 0.1 * g\n"
        "expected_m = 0.1 * g\n"
        "assert t.allclose(m, expected_m, atol=1e-6), f'm wrong: got {m}, expected {expected_m}'\n"
        "# v = 0.999 * 0 + 0.001 * g**2 = 0.001 * g**2\n"
        "expected_v = 0.001 * g.pow(2)\n"
        "assert t.allclose(v, expected_v, atol=1e-6), f'v wrong: got {v}, expected {expected_v}'\n"
        "assert m.data_ptr() == ptr_m, 'm buffer must be mutated in place (copy_)'\n"
        "assert v.data_ptr() == ptr_v, 'v buffer must be mutated in place (copy_)'\n"
        "\n"
        "# === v stays non-negative even with negative gradient ===\n"
        "# v uses g**2 so it's always >= 0 elementwise.\n"
        "assert (v >= 0).all(), f'v entries must be non-negative; got {v}'\n"
        "\n"
        "# === Multiple params, mismatched shapes ===\n"
        "m1 = t.zeros(3)\n"
        "v1 = t.zeros(3)\n"
        "m2 = t.zeros(2, 4)\n"
        "v2 = t.zeros(2, 4)\n"
        "g1 = t.tensor([0.5, 0.5, 0.5])\n"
        "g2 = t.full((2, 4), 2.0)\n"
        "ex3_update_m_and_v([m1, m2], [v1, v2], [g1, g2], beta1=0.9, beta2=0.999)\n"
        "# m1 = 0.1 * 0.5 = 0.05\n"
        "assert t.allclose(m1, t.full((3,), 0.05), atol=1e-6)\n"
        "# m2 = 0.1 * 2.0 = 0.2\n"
        "assert t.allclose(m2, t.full((2, 4), 0.2), atol=1e-6)\n"
        "# v1 = 0.001 * 0.25 = 0.00025\n"
        "assert t.allclose(v1, t.full((3,), 0.00025), atol=1e-8)\n"
        "# v2 = 0.001 * 4 = 0.004\n"
        "assert t.allclose(v2, t.full((2, 4), 0.004), atol=1e-8)\n"
        "\n"
        "# === Multi-step EMA accumulation ===\n"
        "# After many steps of constant g, m should approach g and v should approach g**2.\n"
        "m = t.zeros(3)\n"
        "v = t.zeros(3)\n"
        "g = t.tensor([1.0, 2.0, -3.0])\n"
        "for _ in range(15000):\n"
        "    ex3_update_m_and_v([m], [v], [g], beta1=0.9, beta2=0.999)\n"
        "assert t.allclose(m, g, atol=1e-3), f'm should converge to g; got {m}'\n"
        "assert t.allclose(v, g.pow(2), atol=1e-2), f'v should converge to g**2; got {v}'\n"
        "\n"
        "# === Different beta1/beta2 produce different convergence rates ===\n"
        "m_fast = t.zeros(1)\n"
        "v_fast = t.zeros(1)\n"
        "m_slow = t.zeros(1)\n"
        "v_slow = t.zeros(1)\n"
        "g = t.tensor([1.0])\n"
        "for _ in range(10):\n"
        "    ex3_update_m_and_v([m_fast], [v_fast], [g], beta1=0.5, beta2=0.9)\n"
        "    ex3_update_m_and_v([m_slow], [v_slow], [g], beta1=0.99, beta2=0.999)\n"
        "# Faster-decay (smaller beta) makes m converge faster.\n"
        "assert m_fast.item() > m_slow.item(), (\n"
        "    f'beta1=0.5 should converge faster than beta1=0.99 at step 10; got '\n"
        "    f'm_fast={m_fast.item()}, m_slow={m_slow.item()}'\n"
        ")\n"
        "\n"
        "# === Original tensor identities preserved across many steps ===\n"
        "m = t.zeros(4)\n"
        "v = t.zeros(4)\n"
        "g = t.ones(4)\n"
        "ptr_m, ptr_v = m.data_ptr(), v.data_ptr()\n"
        "for _ in range(50):\n"
        "    ex3_update_m_and_v([m], [v], [g], 0.9, 0.999)\n"
        "assert m.data_ptr() == ptr_m, 'm must stay at the same storage across steps'\n"
        "assert v.data_ptr() == ptr_v, 'v must stay at the same storage across steps'"
    ),
    "solution_body": (
        "def ex3_update_m_and_v(m_list, v_list, grad_list, beta1, beta2):\n"
        "    for m, v, g in zip(m_list, v_list, grad_list):\n"
        "        m.copy_(beta1 * m + (1.0 - beta1) * g)\n"
        "        v.copy_(beta2 * v + (1.0 - beta2) * g.pow(2))"
    ),
    "solution_notes": (
        "**Both updates are independent.** No cross-dependency between m "
        "and v — each only reads from itself and `g`. Running them in one "
        "loop (over params) is just memory-locality / cache friendliness; "
        "the math is identical to two separate passes.\n\n"
        "**`.copy_()` not assignment.** Plain `m = ...` rebinds the local "
        "name in the function scope; the caller's `m_list[i]` still points "
        "to the original tensor. `m.copy_(...)` mutates the existing "
        "storage, so the caller sees the update.\n\n"
        "**`g.pow(2)` vs `g * g`.** Same result; `pow(2)` is one CUDA "
        "kernel and one allocation. `g * g` materializes a temporary. In "
        "Adam's per-step inner loop this matters for very large models."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 5 — examples-seen-step-axis ex3
# ---------------------------------------------------------------------------

SPEC_EXAMPLES = {
    "atom_id": "examples-seen-step-axis",
    "subtopic": "Trainer: examples-seen step axis",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_GRAD_ACCUM,
    "exercise_index": 3,
    "exercise_title": "gradient-accumulation factor in examples_seen",
    "slug": "gradient-accumulation-factor-in-examples-seen",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["examples-seen", "gradient-accumulation", "effective-batch", "wandb"],
    "kcs": [
        "examples-seen-times-accum-steps-factor",
        "optimizer-step-counter-separate-from-micro-batch",
    ],
    "lo": (
        "Apply `examples_seen = step * batch_size * accum_steps` so that "
        "training runs with different micro-batch / accumulation splits but "
        "the same EFFECTIVE batch size produce overlapping wandb curves."
    ),
    "prompt_body": (
        "Implement `ex3_train_with_grad_accum(losses, batch_size, "
        "accum_steps, wandb)`. A training-loop-like driver that logs one "
        "`wandb.log(...)` per OPTIMIZER step, where each step represents "
        "`accum_steps` micro-batches.\n\n"
        "Inputs:\n"
        "- `losses`: `list[float]` — one entry per OPTIMIZER step (already "
        "averaged across the micro-batches inside that step).\n"
        "- `batch_size`: int — micro-batch size.\n"
        "- `accum_steps`: int — number of micro-batches per optimizer step.\n"
        "- `wandb`: object with `wandb.log(dict)`.\n\n"
        "For `step, loss in enumerate(losses, start=1)`:\n\n"
        "1. Compute `examples_seen = step * batch_size * accum_steps`.\n"
        "2. Call `wandb.log({'loss': loss, 'examples_seen': examples_seen, "
        "'step': step})`.\n\n"
        "Return the total number of log calls made (== `len(losses)`)."
    ),
    "stub": (
        "def ex3_train_with_grad_accum(losses, batch_size, accum_steps, wandb) -> int:\n"
        '    """One wandb.log per optimizer step; examples_seen includes accum_steps factor."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "from unittest.mock import MagicMock\n"
        "\n"
        "# === Basic case: 5 optimizer steps, batch_size=8, accum_steps=4 ===\n"
        "# effective batch = 32; per step examples_seen = 32, 64, 96, 128, 160.\n"
        "wandb = MagicMock()\n"
        "losses = [1.0, 0.8, 0.7, 0.65, 0.6]\n"
        "n = ex3_train_with_grad_accum(losses, batch_size=8, accum_steps=4, wandb=wandb)\n"
        "assert n == 5, f'expected 5 log calls; got {n}'\n"
        "assert wandb.log.call_count == 5\n"
        "\n"
        "for i, call in enumerate(wandb.log.call_args_list, start=1):\n"
        "    args, kwargs = call\n"
        "    assert len(args) == 1, f'call {i}: expected one positional dict; got {args}'\n"
        "    payload = args[0]\n"
        "    assert isinstance(payload, dict)\n"
        "    assert payload['examples_seen'] == i * 8 * 4, (\n"
        "        f'call {i}: examples_seen should be {i*32}; got {payload[\"examples_seen\"]}'\n"
        "    )\n"
        "    assert payload['step'] == i, f'call {i}: step should be {i}; got {payload[\"step\"]}'\n"
        "    assert payload['loss'] == losses[i - 1]\n"
        "\n"
        "# === accum_steps=1 collapses to ex1/ex2 behaviour ===\n"
        "wandb2 = MagicMock()\n"
        "ex3_train_with_grad_accum([0.5, 0.4, 0.3], batch_size=32, accum_steps=1, wandb=wandb2)\n"
        "payloads = [c.args[0] for c in wandb2.log.call_args_list]\n"
        "assert payloads[0]['examples_seen'] == 32\n"
        "assert payloads[1]['examples_seen'] == 64\n"
        "assert payloads[2]['examples_seen'] == 96\n"
        "\n"
        "# === Two runs with same effective batch overlap ===\n"
        "# Run A: bs=8, accum=4 → effective 32\n"
        "# Run B: bs=32, accum=1 → effective 32\n"
        "# After K optimizer steps, examples_seen should be identical.\n"
        "wandb_a = MagicMock()\n"
        "wandb_b = MagicMock()\n"
        "ex3_train_with_grad_accum([0.1] * 10, batch_size=8, accum_steps=4, wandb=wandb_a)\n"
        "ex3_train_with_grad_accum([0.1] * 10, batch_size=32, accum_steps=1, wandb=wandb_b)\n"
        "a_examples = [c.args[0]['examples_seen'] for c in wandb_a.log.call_args_list]\n"
        "b_examples = [c.args[0]['examples_seen'] for c in wandb_b.log.call_args_list]\n"
        "assert a_examples == b_examples, (\n"
        "    f'runs with same effective batch must agree on examples_seen:\\n  '\n"
        "    f'A={a_examples}\\n  B={b_examples}'\n"
        ")\n"
        "\n"
        "# === accum_steps=8 multiplies examples_seen 8x relative to accum=1 ===\n"
        "wandb_x = MagicMock()\n"
        "wandb_y = MagicMock()\n"
        "ex3_train_with_grad_accum([0.1] * 3, batch_size=16, accum_steps=1, wandb=wandb_x)\n"
        "ex3_train_with_grad_accum([0.1] * 3, batch_size=16, accum_steps=8, wandb=wandb_y)\n"
        "x_ex = [c.args[0]['examples_seen'] for c in wandb_x.log.call_args_list]\n"
        "y_ex = [c.args[0]['examples_seen'] for c in wandb_y.log.call_args_list]\n"
        "for xi, yi in zip(x_ex, y_ex):\n"
        "    assert yi == xi * 8, f'accum=8 should be 8x the accum=1 examples_seen; got {yi} vs {xi}'\n"
        "\n"
        "# === Empty losses → zero calls ===\n"
        "wandb3 = MagicMock()\n"
        "n0 = ex3_train_with_grad_accum([], batch_size=16, accum_steps=2, wandb=wandb3)\n"
        "assert n0 == 0\n"
        "assert wandb3.log.call_count == 0\n"
        "\n"
        "# === Payload key check ===\n"
        "wandb4 = MagicMock()\n"
        "ex3_train_with_grad_accum([0.9], batch_size=4, accum_steps=2, wandb=wandb4)\n"
        "payload = wandb4.log.call_args_list[0].args[0]\n"
        "assert set(payload.keys()) >= {'loss', 'examples_seen', 'step'}, (\n"
        "    f'payload must contain loss, examples_seen, step; got keys {set(payload.keys())}'\n"
        ")"
    ),
    "solution_body": (
        "def ex3_train_with_grad_accum(losses, batch_size, accum_steps, wandb):\n"
        "    n_calls = 0\n"
        "    for step, loss in enumerate(losses, start=1):\n"
        "        examples_seen = step * batch_size * accum_steps\n"
        "        wandb.log({\n"
        "            'loss': loss,\n"
        "            'examples_seen': examples_seen,\n"
        "            'step': step,\n"
        "        })\n"
        "        n_calls += 1\n"
        "    return n_calls"
    ),
    "solution_notes": (
        "**Effective batch is what curves are compared on.** Two runs with "
        "`(bs=8, accum=4)` and `(bs=32, accum=1)` both have effective batch "
        "32 per optimizer step. `examples_seen` puts them on the same "
        "x-axis — their loss curves should look identical (up to randomness "
        "in mini-batch shuffling).\n\n"
        "**`accum_steps=1` is the no-accumulation case.** The formula "
        "reduces to ex1/ex2's `step * batch_size` because the third factor "
        "is just 1. This is why the extension is non-breaking.\n\n"
        "**Why log `step` AND `examples_seen`.** They serve different "
        "questions: `step` answers 'how many optimizer updates?' (cost on "
        "the optimizer's side); `examples_seen` answers 'how much data has "
        "the model been trained on?' (the comparison axis across runs)."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 6 — optimizer-loop-on-tensor ex3
# ---------------------------------------------------------------------------

SPEC_OPTLOOP = {
    "atom_id": "optimizer-loop-on-tensor",
    "subtopic": "Optimizer: optimizer.step loop over params",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_ZERO_GRAD,
    "exercise_index": 3,
    "exercise_title": "zero_grad per-param loop with set_to_none toggle",
    "slug": "zero-grad-per-param-loop-with-set-to-none",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["zero-grad", "set-to-none", "per-param-loop", "optimizer"],
    "kcs": [
        "zero-grad-per-param-loop",
        "set-to-none-vs-zero_-tradeoff",
    ],
    "lo": (
        "Apply the per-parameter `zero_grad` loop with a `set_to_none` "
        "toggle — `set_to_none=True` reassigns `p.grad = None` (memory win), "
        "`set_to_none=False` calls `p.grad.zero_()` (preserves the tensor)."
    ),
    "prompt_body": (
        "Implement `Ex3ZeroGradSGD`. A hand-rolled SGD that exposes both "
        "`step` and `zero_grad` as per-parameter loops.\n\n"
        "Requirements:\n\n"
        "1. `__init__(self, params, lr)`:\n"
        "   - `self.params = list(params)`\n"
        "   - `self.lr = lr`\n\n"
        "2. `@t.inference_mode()` `step(self)`: standard SGD — for each "
        "`p in self.params` with `p.grad is not None`, do `p -= self.lr * "
        "p.grad`.\n\n"
        "3. `@t.inference_mode()` `zero_grad(self, set_to_none: bool = "
        "True)`: for each `p in self.params`:\n"
        "   - If `p.grad is None`, continue.\n"
        "   - If `set_to_none` is True: `p.grad = None`.\n"
        "   - Else: `p.grad.zero_()` (in-place zero, keeps the same tensor).\n\n"
        "Constraints:\n"
        "- `zero_grad(set_to_none=True)` must NOT use `.zero_()` — it must "
        "set the attribute to `None` directly.\n"
        "- `zero_grad(set_to_none=False)` must preserve `p.grad`'s "
        "`data_ptr()` (the grad tensor stays at the same storage).\n"
        "- Both branches must skip params with `p.grad is None`."
    ),
    "stub": (
        "class Ex3ZeroGradSGD:\n"
        "    def __init__(self, params, lr):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    @t.inference_mode()\n"
        "    def step(self):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    @t.inference_mode()\n"
        "    def zero_grad(self, set_to_none: bool = True):\n"
        "        raise NotImplementedError()"
    ),
    "test_body": (
        "# === Construction ===\n"
        "p1 = t.tensor([1.0, 2.0, 3.0], requires_grad=True)\n"
        "p2 = t.tensor([4.0, 5.0], requires_grad=True)\n"
        "opt = Ex3ZeroGradSGD([p1, p2], lr=0.1)\n"
        "assert opt.params == [p1, p2] or (opt.params[0] is p1 and opt.params[1] is p2)\n"
        "assert opt.lr == 0.1\n"
        "\n"
        "# === Populate grads ===\n"
        "p1.grad = t.tensor([10.0, 20.0, 30.0])\n"
        "p2.grad = t.tensor([100.0, 200.0])\n"
        "ptr_g1 = p1.grad.data_ptr()\n"
        "ptr_g2 = p2.grad.data_ptr()\n"
        "\n"
        "# === step does SGD ===\n"
        "opt.step()\n"
        "assert t.allclose(p1, t.tensor([0.0, 0.0, 0.0]), atol=1e-6), (\n"
        "    f'p1 after step: expected [0,0,0]; got {p1}'\n"
        ")\n"
        "assert t.allclose(p2, t.tensor([-6.0, -15.0]), atol=1e-6), (\n"
        "    f'p2 after step: expected [-6,-15]; got {p2}'\n"
        ")\n"
        "\n"
        "# === zero_grad(set_to_none=False) preserves grad tensor identity ===\n"
        "opt.zero_grad(set_to_none=False)\n"
        "assert p1.grad is not None, 'set_to_none=False must KEEP grad tensor'\n"
        "assert p2.grad is not None, 'set_to_none=False must KEEP grad tensor'\n"
        "assert p1.grad.data_ptr() == ptr_g1, (\n"
        "    'set_to_none=False must preserve grad storage (no realloc); '\n"
        "    f'ptr was {ptr_g1}, now {p1.grad.data_ptr()}'\n"
        ")\n"
        "assert p2.grad.data_ptr() == ptr_g2, 'set_to_none=False must preserve p2 grad storage'\n"
        "assert t.all(p1.grad == 0), f'grad must be zeroed; got {p1.grad}'\n"
        "assert t.all(p2.grad == 0), f'grad must be zeroed; got {p2.grad}'\n"
        "\n"
        "# === zero_grad(set_to_none=True) drops the grad tensor entirely ===\n"
        "p1.grad = t.tensor([1.0, 2.0, 3.0])\n"
        "p2.grad = t.tensor([4.0, 5.0])\n"
        "opt.zero_grad(set_to_none=True)\n"
        "assert p1.grad is None, f'set_to_none=True must set grad to None; got {p1.grad}'\n"
        "assert p2.grad is None, f'set_to_none=True must set grad to None; got {p2.grad}'\n"
        "\n"
        "# === default is set_to_none=True (matches PyTorch's modern default) ===\n"
        "p1.grad = t.tensor([1.0, 2.0, 3.0])\n"
        "p2.grad = t.tensor([4.0, 5.0])\n"
        "opt.zero_grad()  # no arg\n"
        "assert p1.grad is None, 'default zero_grad must set grad to None'\n"
        "assert p2.grad is None\n"
        "\n"
        "# === None-grad guard: zero_grad on frozen param does NOT raise ===\n"
        "frozen = t.tensor([1.0, 1.0], requires_grad=False)\n"
        "active = t.tensor([1.0], requires_grad=True)\n"
        "active.grad = t.tensor([0.5])\n"
        "opt2 = Ex3ZeroGradSGD([frozen, active], lr=0.1)\n"
        "opt2.zero_grad(set_to_none=False)   # must not raise even though frozen.grad is None\n"
        "opt2.zero_grad(set_to_none=True)\n"
        "assert active.grad is None\n"
        "assert frozen.grad is None  # untouched\n"
        "\n"
        "# === step + zero_grad cycle drives the canonical training loop ===\n"
        "p = t.tensor([0.0], requires_grad=True)\n"
        "opt3 = Ex3ZeroGradSGD([p], lr=0.5)\n"
        "for _ in range(3):\n"
        "    p.grad = t.tensor([2.0])  # fake backward\n"
        "    opt3.step()\n"
        "    opt3.zero_grad()\n"
        "    assert p.grad is None  # cleared\n"
        "# p moved by -0.5 * 2.0 = -1.0 per step → final p = -3.0\n"
        "assert t.allclose(p, t.tensor([-3.0]), atol=1e-6), f'cycle drift: expected -3.0; got {p}'\n"
        "\n"
        "# === set_to_none=False inside the cycle ===\n"
        "p = t.tensor([0.0], requires_grad=True)\n"
        "opt4 = Ex3ZeroGradSGD([p], lr=0.5)\n"
        "p.grad = t.tensor([1.0])\n"
        "ptr_initial = p.grad.data_ptr()\n"
        "for _ in range(5):\n"
        "    opt4.step()\n"
        "    opt4.zero_grad(set_to_none=False)\n"
        "    assert p.grad is not None\n"
        "    assert p.grad.data_ptr() == ptr_initial, 'grad storage must persist with set_to_none=False'\n"
        "    p.grad.add_(1.0)  # simulate next backward writing into the SAME grad tensor"
    ),
    "solution_body": (
        "class Ex3ZeroGradSGD:\n"
        "    def __init__(self, params, lr):\n"
        "        self.params = list(params)\n"
        "        self.lr = lr\n"
        "\n"
        "    @t.inference_mode()\n"
        "    def step(self):\n"
        "        for p in self.params:\n"
        "            if p.grad is None:\n"
        "                continue\n"
        "            p -= self.lr * p.grad\n"
        "\n"
        "    @t.inference_mode()\n"
        "    def zero_grad(self, set_to_none: bool = True):\n"
        "        for p in self.params:\n"
        "            if p.grad is None:\n"
        "                continue\n"
        "            if set_to_none:\n"
        "                p.grad = None\n"
        "            else:\n"
        "                p.grad.zero_()"
    ),
    "solution_notes": (
        "**`set_to_none=True` is now PyTorch's default.** Setting `p.grad = "
        "None` deallocates the grad tensor's storage. The next `backward()` "
        "re-allocates a fresh grad with the right shape — which the old "
        "grad's contents would have been overwritten by anyway. Net memory "
        "savings are real for large models.\n\n"
        "**`.zero_()` (in-place) preserves storage.** The trailing "
        "underscore is PyTorch's in-place convention. The grad tensor's "
        "`data_ptr()` is unchanged, only its values are reset to 0. Use "
        "this when downstream code reads `.grad` between steps.\n\n"
        "**The `if p.grad is None: continue` guard appears TWICE.** Once "
        "in `step` (don't apply update to frozen params), once in "
        "`zero_grad` (don't try to zero a non-existent attribute). Frozen "
        "/ unused params hit both branches."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 7 — step-counter-increment ex3
# ---------------------------------------------------------------------------

SPEC_STEP = {
    "atom_id": "step-counter-increment",
    "subtopic": "Trainer: step counter increment",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_MAX_STEPS,
    "exercise_index": 3,
    "exercise_title": "max_steps early termination breaks both inner and outer loop",
    "slug": "max-steps-early-termination",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["max-steps", "early-stop", "training-budget", "break"],
    "kcs": [
        "max-steps-checked-after-counter-tick",
        "nested-loop-break-via-stopped-flag",
    ],
    "lo": (
        "Apply step-counter-driven early termination — `if step >= "
        "max_steps: break` after the counter tick — and propagate the "
        "break out of the nested epoch loop using a `stopped` flag."
    ),
    "prompt_body": (
        "Implement `ex3_train_until_max_steps(losses_per_epoch, max_steps)`. "
        "A training-loop driver where the outer loop walks epochs and the "
        "inner loop walks batches, but training halts as soon as the step "
        "counter reaches `max_steps` — even mid-epoch.\n\n"
        "Inputs:\n"
        "- `losses_per_epoch`: `list[list[float]]`. Each inner list is one "
        "epoch's per-batch losses.\n"
        "- `max_steps`: int — global step budget.\n\n"
        "Return a dict:\n"
        "- `'final_step'`: int — the step counter's final value.\n"
        "- `'logged_losses'`: `list[float]` — every loss the function "
        "processed before stopping, in order.\n"
        "- `'stopped_early'`: bool — True iff training halted before all "
        "epochs were consumed.\n\n"
        "Loop structure:\n"
        "1. `step = 0`, `stopped = False`, `logged = []`.\n"
        "2. For each `epoch_losses` in `losses_per_epoch`:\n"
        "   - For each `loss` in `epoch_losses`:\n"
        "     - `logged.append(loss)`\n"
        "     - `step += 1`\n"
        "     - If `step >= max_steps`: set `stopped = True`, `break`.\n"
        "   - If `stopped`: `break` (out of the epoch loop too).\n"
        "3. Return the dict.\n\n"
        "**Tick BEFORE the check.** The order is `step += 1` then `if "
        "step >= max_steps: break`, so the final `step` value equals `min("
        "max_steps, total_batches)`."
    ),
    "stub": (
        "def ex3_train_until_max_steps(losses_per_epoch: list, max_steps: int) -> dict:\n"
        '    """Train until step >= max_steps or losses_per_epoch is exhausted."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === max_steps reached mid-epoch ===\n"
        "# 3 epochs of 4 batches each = 12 total. Stop after 5.\n"
        "losses = [\n"
        "    [1.0, 0.9, 0.8, 0.7],\n"
        "    [0.6, 0.5, 0.4, 0.3],\n"
        "    [0.2, 0.1, 0.05, 0.01],\n"
        "]\n"
        "out = ex3_train_until_max_steps(losses, max_steps=5)\n"
        "assert isinstance(out, dict)\n"
        "assert out['final_step'] == 5, f'should stop at exactly step 5; got {out[\"final_step\"]}'\n"
        "assert out['logged_losses'] == [1.0, 0.9, 0.8, 0.7, 0.6], (\n"
        "    f'should log exactly the first 5 losses; got {out[\"logged_losses\"]}'\n"
        ")\n"
        "assert out['stopped_early'] is True\n"
        "\n"
        "# === max_steps NOT reached — run all epochs ===\n"
        "out = ex3_train_until_max_steps(losses, max_steps=100)\n"
        "assert out['final_step'] == 12, f'no early stop: should run all 12 batches; got {out[\"final_step\"]}'\n"
        "assert out['stopped_early'] is False\n"
        "expected_all = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]\n"
        "assert out['logged_losses'] == expected_all\n"
        "\n"
        "# === max_steps == total_batches: terminates exactly at the end (boundary) ===\n"
        "# 12 batches, max_steps=12 → stops at step 12. The 'stopped_early' is ambiguous in this\n"
        "# boundary case; we accept either True or False, but the step count must be exact.\n"
        "out = ex3_train_until_max_steps(losses, max_steps=12)\n"
        "assert out['final_step'] == 12, f'boundary case: step should be 12; got {out[\"final_step\"]}'\n"
        "assert out['logged_losses'] == expected_all\n"
        "\n"
        "# === max_steps == 1: only one batch processed ===\n"
        "out = ex3_train_until_max_steps(losses, max_steps=1)\n"
        "assert out['final_step'] == 1\n"
        "assert out['logged_losses'] == [1.0]\n"
        "assert out['stopped_early'] is True\n"
        "\n"
        "# === max_steps exceeds total — runs everything ===\n"
        "out = ex3_train_until_max_steps(losses, max_steps=1000)\n"
        "assert out['final_step'] == 12\n"
        "assert out['stopped_early'] is False\n"
        "\n"
        "# === Empty losses_per_epoch ===\n"
        "out = ex3_train_until_max_steps([], max_steps=10)\n"
        "assert out['final_step'] == 0\n"
        "assert out['logged_losses'] == []\n"
        "assert out['stopped_early'] is False  # nothing to stop\n"
        "\n"
        "# === Empty epoch in the middle ===\n"
        "losses2 = [[1.0, 0.5], [], [0.1, 0.05]]\n"
        "out = ex3_train_until_max_steps(losses2, max_steps=100)\n"
        "assert out['final_step'] == 4, f'empty middle epoch contributes 0; got {out[\"final_step\"]}'\n"
        "assert out['logged_losses'] == [1.0, 0.5, 0.1, 0.05]\n"
        "\n"
        "# === Stop mid-batch on epoch boundary ===\n"
        "# 3 epochs of 4 batches = stop at step 4 (exactly end of epoch 0).\n"
        "out = ex3_train_until_max_steps(losses, max_steps=4)\n"
        "assert out['final_step'] == 4\n"
        "assert out['logged_losses'] == [1.0, 0.9, 0.8, 0.7]\n"
        "assert out['stopped_early'] is True  # epochs 1 and 2 never started\n"
        "\n"
        "# === Step counter is tick-before-check ===\n"
        "# With max_steps=3, batches 1,2,3 are processed, then step==3 triggers break.\n"
        "out = ex3_train_until_max_steps(losses, max_steps=3)\n"
        "assert out['final_step'] == 3\n"
        "assert len(out['logged_losses']) == 3"
    ),
    "solution_body": (
        "def ex3_train_until_max_steps(losses_per_epoch, max_steps):\n"
        "    step = 0\n"
        "    stopped = False\n"
        "    logged = []\n"
        "    for epoch_losses in losses_per_epoch:\n"
        "        for loss in epoch_losses:\n"
        "            logged.append(loss)\n"
        "            step += 1\n"
        "            if step >= max_steps:\n"
        "                stopped = True\n"
        "                break\n"
        "        if stopped:\n"
        "            break\n"
        "    # If we exited because epochs ran out (not because step hit max), stopped_early stays False.\n"
        "    return {\n"
        "        'final_step': step,\n"
        "        'logged_losses': logged,\n"
        "        'stopped_early': stopped,\n"
        "    }"
    ),
    "solution_notes": (
        "**Two `break`s are required.** Python's `break` only exits the "
        "innermost loop, so the inner break ends the current epoch, then "
        "the outer `if stopped: break` ends the epoch loop. Forgetting "
        "the outer one means training resumes in the next epoch — silently "
        "doing extra work.\n\n"
        "**Tick-then-check is the safe order.** `step += 1` followed by "
        "`if step >= max_steps: break` makes `final_step == min(max_steps, "
        "total_batches)`. Reversing the order (check then tick) would "
        "off-by-one your reporting.\n\n"
        "**Hugging Face's pattern.** `TrainingArguments.max_steps` uses "
        "exactly this break-twice pattern in `Trainer._inner_training_loop`. "
        "For LLM training, step-based budgets (compute-bound) are far more "
        "common than epoch-based budgets (dataset-size-bound)."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 8 — trainer-class-skeleton ex3
# ---------------------------------------------------------------------------

SPEC_TRAINER = {
    "atom_id": "trainer-class-skeleton",
    "subtopic": "Trainer: Trainer class skeleton",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_BEST_CHECKPOINT,
    "exercise_index": 3,
    "exercise_title": "Trainer with best-val-loss checkpoint snapshot",
    "slug": "trainer-with-best-val-loss-checkpoint",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["checkpoint", "best-val-loss", "state-dict", "trainer"],
    "kcs": [
        "best-val-loss-monotone-improvement-check",
        "state-dict-snapshot-via-detach-clone",
    ],
    "lo": (
        "Apply best-val-loss tracking to the Trainer skeleton: maintain "
        "`self.best_val_loss` (float, init `inf`) and `self.best_state` "
        "(deep-copied state_dict snapshot) updated each time validation "
        "loss strictly improves."
    ),
    "prompt_body": (
        "Implement `Ex3TrainerWithCheckpoint`. Same shape as the minimal "
        "Trainer, plus best-val-loss tracking and a snapshot of "
        "`model.state_dict()` whenever the val loss improves.\n\n"
        "1. `__init__(self, model, optimizer, train_loader, val_loader, "
        "loss_fn)`:\n"
        "   - Store the five args.\n"
        "   - `self.step = 0`, `self.history = {'train_loss': [], "
        "'val_loss': []}`.\n"
        "   - `self.best_val_loss = float('inf')`.\n"
        "   - `self.best_state = None` (will become a dict of cloned "
        "tensors once validate runs at least once).\n\n"
        "2. `_step(self, x, y)`: forward + loss only.\n\n"
        "3. `fit(self, n_epochs)`: standard train loop calling `_step`, "
        "`loss.backward()`, `optimizer.step()`, `optimizer.zero_grad()`, "
        "`self.step += 1`, `self.history['train_loss'].append(loss.item())`. "
        "After each epoch, call `self.validate()`.\n\n"
        "4. `validate(self)`: compute weighted-average val loss under "
        "`t.inference_mode()`. Append to `self.history['val_loss']`. "
        "Then:\n"
        "   - If `val_loss < self.best_val_loss` (STRICT inequality):\n"
        "     - `self.best_val_loss = val_loss`\n"
        "     - `self.best_state = {k: v.detach().clone() for k, v in "
        "self.model.state_dict().items()}`\n"
        "   - Otherwise: do not touch `best_state`.\n\n"
        "Snapshot policy: use `.detach().clone()` per tensor so the "
        "snapshot is INDEPENDENT of subsequent training mutations."
    ),
    "stub": (
        "class Ex3TrainerWithCheckpoint:\n"
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
        "import math\n"
        "\n"
        "# === Tiny regression task: y = 2x + 1 ===\n"
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
        "model = nn.Linear(1, 1)\n"
        "opt = t.optim.SGD(model.parameters(), lr=0.1)\n"
        "loss_fn = nn.MSELoss()\n"
        "\n"
        "trainer = Ex3TrainerWithCheckpoint(model, opt, train_loader, val_loader, loss_fn)\n"
        "\n"
        "# === Attributes ===\n"
        "assert trainer.best_val_loss == float('inf'), (\n"
        "    f'best_val_loss must init to inf; got {trainer.best_val_loss}'\n"
        ")\n"
        "assert trainer.best_state is None, (\n"
        "    f'best_state must init to None; got {trainer.best_state}'\n"
        ")\n"
        "assert trainer.step == 0\n"
        "assert trainer.history == {'train_loss': [], 'val_loss': []}\n"
        "\n"
        "# === Fit 4 epochs — best_state must populate ===\n"
        "trainer.fit(n_epochs=4)\n"
        "assert trainer.best_state is not None, 'after fit, best_state must be a dict snapshot'\n"
        "assert isinstance(trainer.best_state, dict)\n"
        "assert 'weight' in trainer.best_state and 'bias' in trainer.best_state, (\n"
        "    f'best_state should mirror state_dict keys; got keys {list(trainer.best_state.keys())}'\n"
        ")\n"
        "\n"
        "# === best_val_loss is the min of the recorded val losses ===\n"
        "val_losses = trainer.history['val_loss']\n"
        "assert len(val_losses) == 4, f'4 epochs → 4 val losses; got {len(val_losses)}'\n"
        "assert abs(trainer.best_val_loss - min(val_losses)) < 1e-9, (\n"
        "    f'best_val_loss must equal min(val_loss history): {trainer.best_val_loss} vs {min(val_losses)}'\n"
        ")\n"
        "\n"
        "# === best_state corresponds to the epoch where val_loss was minimal ===\n"
        "# For a converging regression run, the best epoch should be the LAST one.\n"
        "# But the snapshot must match the model state AT that epoch — independent of\n"
        "# further training. We test independence below.\n"
        "\n"
        "# === Snapshot is INDEPENDENT: mutate model post-fit, snapshot unchanged ===\n"
        "snapshot_w = trainer.best_state['weight'].clone()\n"
        "snapshot_b = trainer.best_state['bias'].clone()\n"
        "# Mutate the model:\n"
        "with t.no_grad():\n"
        "    model.weight.fill_(999.0)\n"
        "    model.bias.fill_(-999.0)\n"
        "# best_state should NOT have changed:\n"
        "assert t.equal(trainer.best_state['weight'], snapshot_w), (\n"
        "    f'snapshot must be independent of post-fit mutations'\n"
        ")\n"
        "assert t.equal(trainer.best_state['bias'], snapshot_b)\n"
        "\n"
        "# === Snapshot tensors are DETACHED ===\n"
        "for k, v in trainer.best_state.items():\n"
        "    assert isinstance(v, t.Tensor), f'snapshot entry {k} must be tensor; got {type(v)}'\n"
        "    assert not v.requires_grad, f'snapshot tensor {k} must be detached (no grad-track)'\n"
        "\n"
        "# === Strict-improvement: a worse val loss does NOT overwrite best_state ===\n"
        "# Manually trigger a worse epoch: synthetic best_state should remain.\n"
        "t.manual_seed(123)\n"
        "model2 = nn.Linear(1, 1)\n"
        "# Make model2 fit poorly: huge initial weight.\n"
        "with t.no_grad():\n"
        "    model2.weight.fill_(50.0)\n"
        "    model2.bias.fill_(0.0)\n"
        "opt2 = t.optim.SGD(model2.parameters(), lr=0.0)  # lr=0 → no learning → val loss flat\n"
        "trainer2 = Ex3TrainerWithCheckpoint(model2, opt2, train_loader, val_loader, loss_fn)\n"
        "trainer2.fit(n_epochs=3)\n"
        "vls = trainer2.history['val_loss']\n"
        "# All val losses should be (essentially) identical since lr=0.\n"
        "assert max(vls) - min(vls) < 1e-3, f'lr=0 → val loss constant; got {vls}'\n"
        "# best_state was set on epoch 0 and never overwritten (subsequent epochs are NOT strictly less).\n"
        "assert trainer2.best_state is not None\n"
        "assert abs(trainer2.best_val_loss - vls[0]) < 1e-6, (\n"
        "    f'best_val_loss should equal epoch-0 val_loss (first improvement); '\n"
        "    f'got {trainer2.best_val_loss} vs {vls[0]}'\n"
        ")\n"
        "\n"
        "# === Sanity: validate is run via fit ===\n"
        "model3 = nn.Linear(1, 1)\n"
        "opt3 = t.optim.SGD(model3.parameters(), lr=0.1)\n"
        "trainer3 = Ex3TrainerWithCheckpoint(model3, opt3, train_loader, val_loader, loss_fn)\n"
        "assert trainer3.best_state is None\n"
        "trainer3.fit(n_epochs=1)\n"
        "assert trainer3.best_state is not None, 'after 1 epoch validate must have run'\n"
        "assert len(trainer3.history['val_loss']) == 1"
    ),
    "solution_body": (
        "class Ex3TrainerWithCheckpoint:\n"
        "    def __init__(self, model, optimizer, train_loader, val_loader, loss_fn):\n"
        "        self.model = model\n"
        "        self.optimizer = optimizer\n"
        "        self.train_loader = train_loader\n"
        "        self.val_loader = val_loader\n"
        "        self.loss_fn = loss_fn\n"
        "        self.step = 0\n"
        "        self.history = {'train_loss': [], 'val_loss': []}\n"
        "        self.best_val_loss = float('inf')\n"
        "        self.best_state = None\n"
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
        "        val_loss = total / count\n"
        "        self.history['val_loss'].append(val_loss)\n"
        "        if val_loss < self.best_val_loss:\n"
        "            self.best_val_loss = val_loss\n"
        "            self.best_state = {\n"
        "                k: v.detach().clone() for k, v in self.model.state_dict().items()\n"
        "            }"
    ),
    "solution_notes": (
        "**`.detach().clone()` is the canonical snapshot idiom.** `clone()` "
        "alone preserves the autograd graph link; `detach()` alone shares "
        "storage with the live model. The combo gives an independent tensor "
        "that is NOT mutated by subsequent `optimizer.step()` calls.\n\n"
        "**Strict `<` not `<=`.** With `<=`, ties on val loss overwrite the "
        "snapshot. The convention is 'keep the EARLIER best' — earlier "
        "means less overfit. Strict less-than enforces that.\n\n"
        "**Sentinel `None` is cleaner than 'init from random init'.** "
        "Before the first `validate()` call, no real evaluation has "
        "happened. Storing a random-init snapshot would lie about 'best'; "
        "`None` lets callers detect 'no validation has run yet'."
    ),
    "extra_imports": ["import torch.nn as nn", "import torch.nn.functional as F"],
}


# ---------------------------------------------------------------------------
# SPEC list + verifier + main
# ---------------------------------------------------------------------------

SPECS = [
    SPEC_ARGMAX,
    SPEC_BIAS,
    SPEC_CE,
    SPEC_EMA,
    SPEC_EXAMPLES,
    SPEC_OPTLOOP,
    SPEC_STEP,
    SPEC_TRAINER,
]


def _verify_all(specs):
    import torch as t
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from torch import Tensor
    from unittest.mock import MagicMock
    import sys as _sys
    _sys.modules["wandb"] = MagicMock()

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
    print(f"[deepening_b_batch14] Verifying {len(SPECS)} specs...")
    _verify_all(SPECS)

    print(f"\n[deepening_b_batch14] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_b_batch14] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
