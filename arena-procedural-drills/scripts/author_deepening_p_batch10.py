#!/usr/bin/env python3
"""Author ex2 deepening drills (batch10) for 5 custom-Tensor + 3 DCGAN atoms.

Each ex2 targets a DISTINCT facet beyond ex1 — not a repackaged ex1.

Custom-tensor atoms (need MiniTensor + Recipe preamble):
  * linear-affine-on-custom-tensor   — ex2 batch-summed bias backward
  * logsumexp-cross-entropy          — ex2 gradient = (softmax - onehot) / B
  * module-base-class-custom         — ex2 train()/eval() recursive toggle
  * parameter-wrap-around-tensor     — ex2 fix anti-pattern by re-wrapping IS-A
  * sgd-vanilla-from-scratch         — ex2 multi-step quadratic, loss monotone

DCGAN atoms (plain pytorch):
  * bce-log-loss-real-fake           — ex2 logits-form via BCE-with-logits
  * bn-weight-bias-init-pattern      — ex2 near-identity at init (gamma~1,beta=0)
  * channel-list-reverse-build       — ex2 assemble decoder Sequential, verify spatial doubling
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone


_CUSTOM_TENSOR_PREAMBLE = (
    "# === manual autograd primitives — shared across all drills in this folder ===\n"
    "from dataclasses import dataclass, field\n"
    "from typing import Any, Callable, Optional\n"
    "\n"
    "grad_tracking_enabled = True\n"
    "\n"
    "@dataclass\n"
    "class Recipe:\n"
    "    func: Optional[Callable] = None\n"
    "    args: tuple = ()\n"
    "    kwargs: dict = field(default_factory=dict)\n"
    "    parents: dict = field(default_factory=dict)\n"
    "\n"
    "class MiniTensor:\n"
    "    def __init__(self, array, requires_grad: bool = False, recipe=None):\n"
    "        self.array = array\n"
    "        self.requires_grad = requires_grad\n"
    "        self.recipe = recipe\n"
    "        self.grad = None\n"
    "    def __repr__(self):\n"
    "        return f'MiniTensor({self.array!r}, requires_grad={self.requires_grad})'"
)


def _spec(
    *,
    atom_id: str,
    subtopic: str,
    topic_folder: str,
    recap: str,
    ex_title: str,
    slug: str,
    bloom: str,
    difficulty_num: int,
    keywords: list[str],
    kcs: list[str],
    lo: str,
    prompt_body: str,
    stub: str,
    test_body: str,
    solution_body: str,
    solution_notes: str = "",
    extra_imports: list[str] | None = None,
    needs_custom_tensor: bool = False,
) -> dict:
    dots = ("🔴" * difficulty_num) + ("⚪" * (5 - difficulty_num))
    extras = list(extra_imports or [])
    if needs_custom_tensor:
        extras = [_CUSTOM_TENSOR_PREAMBLE, *extras]
    return {
        "atom_id": atom_id,
        "subtopic": subtopic,
        "topic_folder": topic_folder,
        "atom_recap_md": recap,
        "exercise_index": 2,
        "exercise_title": ex_title,
        "slug": slug,
        "bloom_level": bloom,
        "difficulty_num": difficulty_num,
        "difficulty_dots": dots,
        "keywords": keywords,
        "kcs": kcs,
        "lo": lo,
        "prompt_body": prompt_body,
        "stub": stub,
        "test_body": test_body,
        "solution_body": solution_body,
        "solution_notes": solution_notes,
        "extra_imports": extras,
    }


# =========================================================================
# 1. linear-affine-on-custom-tensor ex2: bias backward = sum over batch axis
# =========================================================================

SPEC_LINEAR_BIAS_BACK = _spec(
    atom_id="linear-affine-on-custom-tensor",
    subtopic="Backprop: Linear affine on custom Tensor",
    topic_folder="prereqs_custom_tensor",
    needs_custom_tensor=True,
    recap=(
        "## Linear-affine backward — bias path — deepening\n"
        "\n"
        "Forward: `out = x @ W + b` broadcasts `b: (out_f,)` over the batch "
        "axis to produce `out: (B, out_f)`. Backward through that broadcast "
        "is the **inverse** operation: an upstream gradient `grad_out` of "
        "shape `(B, out_f)` must collapse back to `(out_f,)` for `b`'s grad.\n"
        "\n"
        "The collapse rule is: **sum over every axis that got broadcast.** "
        "Here that's the leading batch axis:\n"
        "\n"
        "```python\n"
        "grad_b = grad_out.sum(dim=0)   # (B, out_f) -> (out_f,)\n"
        "```\n"
        "\n"
        "If `b` had been `(1, out_f)` instead (keepdim broadcast), the rule "
        "would be `grad_out.sum(dim=0, keepdim=True)` — preserve the same "
        "shape the forward broadcast started from."
    ),
    ex_title="bias backward for Linear: unbroadcast grad over batch axis",
    slug="linear-bias-backward-unbroadcast",
    bloom="Apply",
    difficulty_num=3,
    keywords=["linear", "bias-backward", "unbroadcast", "sum-over-batch"],
    kcs=["linear-affine-on-custom-tensor", "unbroadcast-via-sum"],
    lo=(
        "Apply the unbroadcast-via-sum rule to collapse an upstream "
        "(B, out_f) gradient back to (out_f,) — the bias-backward for an "
        "affine Linear layer over MiniTensors."
    ),
    prompt_body=(
        "Implement `linear_bias_backward(grad_out, bias)`. This is the "
        "**reverse-pass contribution to the bias** in the affine map "
        "`out = mm + bias`.\n\n"
        "Inputs:\n"
        "- `grad_out`: a raw `torch.Tensor` of shape `(B, out_f)` — the "
        "upstream gradient flowing into `out`.\n"
        "- `bias`:     a `MiniTensor` of shape `(out_f,)` — the parameter "
        "we're computing grad for (passed only for shape reference).\n\n"
        "Behavior:\n"
        "1. Sum `grad_out` over the batch axis (axis 0). The bias broadcast "
        "over the batch, so the backward collapses over the batch.\n"
        "2. Confirm the result shape equals `bias.array.shape` — if it "
        "doesn't, your unbroadcast is wrong.\n"
        "3. Return the raw `torch.Tensor` of shape `(out_f,)`.\n\n"
        "Return type is a plain tensor, NOT a MiniTensor — backward "
        "contributions are raw tensors that the reverse-pass dispatcher "
        "accumulates into a `grads` dict.\n\n"
        "**Why this is the deepening of ex1.** Ex1 built the forward "
        "`mm + bias` Recipe chain; ex2 builds the backward primitive that "
        "the dispatcher will pair with that Recipe. Together they make a "
        "fully end-to-end-trainable Linear."
    ),
    stub=(
        "def linear_bias_backward(grad_out, bias: MiniTensor):\n"
        '    """Collapse a (B, out_f) grad to (out_f,) by summing the batch axis."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- invariant 1: shape collapse (B, out_f) -> (out_f,) ---\n"
        "B, out_f = 4, 5\n"
        "bias = MiniTensor(t.zeros(out_f), requires_grad=True)\n"
        "grad_out = t.ones(B, out_f)\n"
        "gb = linear_bias_backward(grad_out, bias)\n"
        "assert isinstance(gb, t.Tensor), f'must return raw torch.Tensor, got {type(gb).__name__}'\n"
        "assert gb.shape == (out_f,), f'shape: {gb.shape}'\n"
        "\n"
        "# --- invariant 2: ones-input collapses to all-B value (sum, not mean) ---\n"
        "assert t.allclose(gb, t.full((out_f,), float(B))), (\n"
        "    f'sum-over-batch of all-ones must equal B for each out-channel, got {gb}'\n"
        ")\n"
        "\n"
        "# --- invariant 3: matches torch.autograd reference on a real affine forward ---\n"
        "x_ref = t.randn(B, 3)\n"
        "w_ref = t.randn(3, out_f, requires_grad=True)\n"
        "b_ref = t.randn(out_f, requires_grad=True)\n"
        "out_ref = x_ref @ w_ref + b_ref\n"
        "grad_seed = t.randn(B, out_f)\n"
        "out_ref.backward(grad_seed)\n"
        "ours = linear_bias_backward(grad_seed, MiniTensor(b_ref.detach()))\n"
        "assert t.allclose(ours, b_ref.grad, atol=1e-5), (\n"
        "    f'must match torch autograd on the same seed; ours={ours} vs ref={b_ref.grad}'\n"
        ")\n"
        "\n"
        "# --- invariant 4: non-trivial batch size, dtype preserved ---\n"
        "g64 = t.randn(32, 10, dtype=t.float64)\n"
        "b64 = MiniTensor(t.zeros(10, dtype=t.float64))\n"
        "gb64 = linear_bias_backward(g64, b64)\n"
        "assert gb64.shape == (10,) and gb64.dtype == t.float64, f'dtype/shape: {gb64.dtype}/{gb64.shape}'\n"
        "assert t.allclose(gb64, g64.sum(dim=0))"
    ),
    solution_body=(
        "def linear_bias_backward(grad_out, bias: MiniTensor):\n"
        "    # bias broadcast over axis 0 in the forward -> sum over axis 0 in reverse\n"
        "    gb = grad_out.sum(dim=0)\n"
        "    assert gb.shape == bias.array.shape, (\n"
        "        f'bias-grad shape {gb.shape} must equal bias shape {bias.array.shape}'\n"
        "    )\n"
        "    return gb"
    ),
    solution_notes=(
        "**Unbroadcast = sum over expanded axes.** The general rule: if the "
        "forward used `a + b` where `b` was implicitly expanded along axes "
        "`E`, the backward to `b` is `grad_out.sum(dim=E)`. For Linear's "
        "bias the expansion is the batch axis, so `sum(dim=0)` recovers the "
        "right shape.\n\n"
        "**Why the shape assert.** Catching a shape mismatch at the "
        "backward boundary is the cheapest debugging — most autograd bugs "
        "show up as 'shape (B, out_f) cannot be assigned to bias.grad of "
        "shape (out_f,)' three frames deeper. Failing fast here saves "
        "30 seconds of stack-walking."
    ),
)


# =========================================================================
# 2. logsumexp-cross-entropy ex2: gradient = (softmax - onehot) / B
# =========================================================================

SPEC_LOGSUMEXP_GRAD = _spec(
    atom_id="logsumexp-cross-entropy",
    subtopic="Loss: logsumexp cross-entropy",
    topic_folder="prereqs_custom_tensor",
    needs_custom_tensor=True,
    recap=(
        "## Cross-entropy gradient identity — deepening\n"
        "\n"
        "The gradient of cross-entropy w.r.t. the logits has a closed form "
        "that makes the backward pass trivial — no autograd machinery "
        "needed:\n"
        "\n"
        "```\n"
        "dL/d(logits[i, k]) = (softmax(logits[i])[k] - onehot(target[i])[k]) / B\n"
        "```\n"
        "\n"
        "for the mean-reduced CE loss. Equivalently in tensor form:\n"
        "\n"
        "```python\n"
        "probs = softmax(logits, dim=-1)\n"
        "onehot = F.one_hot(target, num_classes=C).float()\n"
        "grad_logits = (probs - onehot) / B\n"
        "```\n"
        "\n"
        "**Why this is the killer identity.** The naive chain rule routes "
        "through logsumexp and the per-sample picker — both differentiable, "
        "but you'd have a 3-step backward graph. The closed form collapses "
        "to a single op, which is what `nn.CrossEntropyLoss.backward` "
        "actually does internally."
    ),
    ex_title="closed-form CE gradient: (softmax - onehot) / B",
    slug="cross-entropy-grad-softmax-minus-onehot",
    bloom="Apply",
    difficulty_num=3,
    keywords=["cross-entropy", "gradient", "softmax", "onehot", "closed-form"],
    kcs=["logsumexp-cross-entropy", "softmax-minus-onehot-grad"],
    lo=(
        "Apply the closed-form identity `dL/dlogits = (softmax(logits) - "
        "onehot(target)) / B` to compute the cross-entropy gradient in one "
        "step, then validate against torch.autograd as the witness."
    ),
    prompt_body=(
        "Implement `cross_entropy_grad(logits, target)`. The gradient of "
        "mean-reduced cross-entropy w.r.t. the logits, in CLOSED FORM:\n\n"
        "```\n"
        "grad[i, k] = (softmax(logits[i])[k] - 1{k == target[i]}) / B\n"
        "```\n\n"
        "Inputs:\n"
        "- `logits`: shape `(B, C)`, float.\n"
        "- `target`: shape `(B,)`, integer class indices in `[0, C)`.\n\n"
        "Output: shape `(B, C)`, float — same shape as `logits`.\n\n"
        "Recipe:\n"
        "1. `probs = t.softmax(logits, dim=-1)` — shape `(B, C)`.\n"
        "2. `onehot = F.one_hot(target, num_classes=logits.shape[1]).float()` "
        "— shape `(B, C)`.\n"
        "3. `grad = (probs - onehot) / B`.\n"
        "4. Return `grad`.\n\n"
        "**Do NOT call `.backward()`** on a torch loss to compute this — the "
        "drill is the closed form. The test cell cross-checks against "
        "autograd as a witness (which you call from the TEST, not from your "
        "implementation)."
    ),
    stub=(
        "def cross_entropy_grad(logits, target):\n"
        '    """Closed-form gradient of mean-reduced CE: (softmax - onehot) / B."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "import torch.nn.functional as F\n"
        "# --- invariant 1: shape matches logits ---\n"
        "logits = t.tensor([[2.0, 1.0, 0.1], [0.5, -1.0, 3.0]])\n"
        "target = t.tensor([0, 2])\n"
        "g = cross_entropy_grad(logits, target)\n"
        "assert g.shape == logits.shape, f'shape: {g.shape}'\n"
        "\n"
        "# --- invariant 2: matches autograd witness on a random batch ---\n"
        "t.manual_seed(0)\n"
        "L = t.randn(8, 5, requires_grad=True)\n"
        "T = t.randint(0, 5, (8,))\n"
        "loss = F.cross_entropy(L, T)\n"
        "loss.backward()\n"
        "ours = cross_entropy_grad(L.detach(), T)\n"
        "assert t.allclose(ours, L.grad, atol=1e-6), (\n"
        "    f'closed-form must match torch autograd; max diff = {(ours - L.grad).abs().max().item()}'\n"
        ")\n"
        "\n"
        "# --- invariant 3: gradient sums to zero per-row when divided by B ---\n"
        "# softmax sums to 1, onehot sums to 1 -> their diff sums to 0 per row.\n"
        "row_sums = g.sum(dim=-1)\n"
        "assert t.allclose(row_sums, t.zeros_like(row_sums), atol=1e-6), (\n"
        "    f'each row of the gradient must sum to 0 (softmax-onehot cancellation), got {row_sums}'\n"
        ")\n"
        "\n"
        "# --- invariant 4: sign at the target class is NEGATIVE (push logit UP) ---\n"
        "# At index target[i], grad = (prob - 1) / B < 0 since prob in (0, 1).\n"
        "B = logits.shape[0]\n"
        "for i in range(B):\n"
        "    assert g[i, target[i].item()].item() < 0, (\n"
        "        f'grad at target index must be negative (push logit up), got {g[i, target[i].item()]}'\n"
        "    )\n"
        "\n"
        "# --- invariant 5: scale is 1/B (mean-reduction, not sum) ---\n"
        "# Same logits, twice the batch -> grad magnitude halves (per element).\n"
        "L_small = t.randn(4, 3, requires_grad=True)\n"
        "T_small = t.tensor([0, 1, 2, 0])\n"
        "L_big = L_small.detach().repeat(2, 1).requires_grad_(True)\n"
        "T_big = T_small.repeat(2)\n"
        "g_small = cross_entropy_grad(L_small.detach(), T_small)\n"
        "g_big = cross_entropy_grad(L_big.detach(), T_big)\n"
        "# Each element of g_big should be half the corresponding element of g_small (tiled).\n"
        "assert t.allclose(g_big[:4], g_small / 2, atol=1e-6), (\n"
        "    f'doubling batch should halve the per-element gradient (mean-reduction): {g_big[:4]} vs {g_small / 2}'\n"
        ")"
    ),
    solution_body=(
        "def cross_entropy_grad(logits, target):\n"
        "    import torch.nn.functional as F\n"
        "    B, C = logits.shape\n"
        "    probs = t.softmax(logits, dim=-1)               # (B, C)\n"
        "    onehot = F.one_hot(target, num_classes=C).to(probs.dtype)  # (B, C)\n"
        "    return (probs - onehot) / B"
    ),
    solution_notes=(
        "**Why the closed form exists.** Cross-entropy + softmax is the "
        "canonical 'exponential-family + log-link' pair from generalized "
        "linear models. For any such pair the gradient of the negative "
        "log-likelihood w.r.t. the natural parameter (here, logits) is "
        "`predicted - observed` — i.e. `softmax(logits) - onehot(target)`. "
        "The `/B` comes purely from mean-reduction.\n\n"
        "**Sign convention.** `grad[i, target[i]] < 0` says SGD will push "
        "the target logit UP (because `p - lr * grad` adds magnitude). "
        "Conversely `grad[i, k] > 0` for non-target classes pushes those "
        "logits DOWN. Net effect: each step nudges the probabilities "
        "toward the one-hot target.\n\n"
        "**Numerical advantage.** Computing this directly bypasses the "
        "logsumexp + arange-fancy-index path entirely. Frameworks "
        "(`nn.CrossEntropyLoss`) fuse logsumexp-forward with "
        "softmax-minus-onehot-backward — never building the full graph."
    ),
)


# =========================================================================
# 3. module-base-class-custom ex2: train()/eval() recursive toggle
# =========================================================================

SPEC_MODULE_TRAIN_EVAL = _spec(
    atom_id="module-base-class-custom",
    subtopic="Backprop: Module base class custom",
    topic_folder="prereqs_custom_tensor",
    needs_custom_tensor=True,
    recap=(
        "## Module.train() / eval() — recursive toggle — deepening\n"
        "\n"
        "Beyond the `__setattr__`-as-registrar pattern, a Module base class "
        "needs ONE more piece: a recursive `.train(mode)` / `.eval()` "
        "toggle that flips a `.training` flag on EVERY submodule.\n"
        "\n"
        "```python\n"
        "class Module:\n"
        "    def __init__(self):\n"
        "        ...\n"
        "        object.__setattr__(self, 'training', True)\n"
        "    def train(self, mode: bool = True):\n"
        "        self.training = mode\n"
        "        for m in self._modules.values():\n"
        "            m.train(mode)\n"
        "        return self\n"
        "    def eval(self):\n"
        "        return self.train(False)\n"
        "```\n"
        "\n"
        "**Why every submodule.** BatchNorm and Dropout in nested layers "
        "read `self.training` to decide their forward behavior. A single "
        "`model.eval()` call must reach every leaf — otherwise a deeply-"
        "nested BN keeps updating running stats during inference. Bug.\n"
        "\n"
        "**Returns `self`** so you can chain: `model.eval().to(device)`."
    ),
    ex_title="recursive train()/eval() toggle propagates .training to every submodule",
    slug="module-train-eval-recursive-toggle",
    bloom="Apply",
    difficulty_num=3,
    keywords=["module", "train-mode", "eval-mode", "recursive", "training-flag"],
    kcs=["module-base-class-custom", "module-train-eval-toggle"],
    lo=(
        "Apply the recursive `.train(mode)` pattern over a custom Module "
        "base class so that toggling the root module's training state "
        "propagates to every directly-assigned submodule and transitively."
    ),
    prompt_body=(
        "Extend the minimal `Module` base class from ex1 with a "
        "`.training` flag and `.train(mode)` / `.eval()` methods that "
        "recursively toggle every submodule.\n\n"
        "Required surface (re-define both `Parameter` and `Module` here so "
        "the drill is self-contained):\n\n"
        "**1. `Parameter(MiniTensor)`** — same as ex1.\n\n"
        "**2. `Module` base class with:**\n"
        "   - `__init__(self)` — initializes `_parameters = {}`, "
        "`_modules = {}`, AND `training = True` (all via `object.__setattr__` "
        "to bypass the custom `__setattr__`).\n"
        "   - `__setattr__` — same registrar pattern as ex1.\n"
        "   - `parameters(self)` — same recursive walker as ex1.\n"
        "   - **`train(self, mode=True)`** — set `self.training = mode`, "
        "then call `m.train(mode)` for every `m in self._modules.values()`. "
        "Return `self`.\n"
        "   - **`eval(self)`** — call `self.train(False)` and return its "
        "result (= self).\n"
        "   - `forward(self, *args, **kwargs)` — raise NotImplementedError.\n\n"
        "**Don't forget to bootstrap `training` in `__init__`.** Using "
        "`self.training = True` would route through the custom "
        "`__setattr__`, which expects `_parameters` to already exist. Use "
        "`object.__setattr__(self, 'training', True)`.\n\n"
        "**Don't forget to return self.** Both `train()` and `eval()` must "
        "return `self` so the chaining idiom `model.eval()(x)` works."
    ),
    stub=(
        "class Parameter(MiniTensor):\n"
        "    def __init__(self, array, requires_grad: bool = True):\n"
        "        super().__init__(array, requires_grad=requires_grad)\n"
        "\n"
        "\n"
        "class Module:\n"
        '    """Module with recursive train()/eval() toggle."""\n'
        "    def __init__(self):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    def __setattr__(self, name, value):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    def parameters(self):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    def train(self, mode: bool = True):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    def eval(self):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    def forward(self, *args, **kwargs):\n"
        "        raise NotImplementedError()"
    ),
    test_body=(
        "# --- invariant 1: fresh module starts in training mode ---\n"
        "class TinyLayer(Module):\n"
        "    def __init__(self, in_f, out_f):\n"
        "        super().__init__()\n"
        "        self.weight = Parameter(t.zeros(in_f, out_f))\n"
        "        self.bias = Parameter(t.zeros(out_f))\n"
        "\n"
        "class Net(Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.layer1 = TinyLayer(3, 5)\n"
        "        self.layer2 = TinyLayer(5, 2)\n"
        "\n"
        "net = Net()\n"
        "assert net.training is True, 'fresh module must start in training mode'\n"
        "assert net.layer1.training is True and net.layer2.training is True\n"
        "\n"
        "# --- invariant 2: eval() propagates to every submodule ---\n"
        "ret = net.eval()\n"
        "assert ret is net, 'eval() must return self for chaining'\n"
        "assert net.training is False\n"
        "assert net.layer1.training is False, 'eval must propagate to layer1'\n"
        "assert net.layer2.training is False, 'eval must propagate to layer2'\n"
        "\n"
        "# --- invariant 3: train() restores all submodules ---\n"
        "ret2 = net.train()\n"
        "assert ret2 is net, 'train() must return self for chaining'\n"
        "assert net.training is True\n"
        "assert net.layer1.training is True and net.layer2.training is True\n"
        "\n"
        "# --- invariant 4: deeper nesting — three levels ---\n"
        "class Block(Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.inner = TinyLayer(4, 4)\n"
        "\n"
        "class Stack(Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.b1 = Block()\n"
        "        self.b2 = Block()\n"
        "\n"
        "stack = Stack()\n"
        "stack.eval()\n"
        "# Every leaf must have training=False, across 3 levels.\n"
        "assert stack.training is False\n"
        "assert stack.b1.training is False and stack.b2.training is False\n"
        "assert stack.b1.inner.training is False, 'must reach 3-deep nested leaf'\n"
        "assert stack.b2.inner.training is False\n"
        "\n"
        "# --- invariant 5: train(False) is alias for eval() ---\n"
        "stack.train(True)\n"
        "assert stack.training is True and stack.b1.inner.training is True\n"
        "stack.train(False)\n"
        "assert stack.training is False and stack.b1.inner.training is False"
    ),
    solution_body=(
        "class Parameter(MiniTensor):\n"
        "    def __init__(self, array, requires_grad: bool = True):\n"
        "        super().__init__(array, requires_grad=requires_grad)\n"
        "\n"
        "\n"
        "class Module:\n"
        "    def __init__(self):\n"
        "        object.__setattr__(self, '_parameters', {})\n"
        "        object.__setattr__(self, '_modules', {})\n"
        "        object.__setattr__(self, 'training', True)\n"
        "\n"
        "    def __setattr__(self, name, value):\n"
        "        if isinstance(value, Parameter):\n"
        "            self._parameters[name] = value\n"
        "            self._modules.pop(name, None)\n"
        "        elif isinstance(value, Module):\n"
        "            self._modules[name] = value\n"
        "            self._parameters.pop(name, None)\n"
        "        else:\n"
        "            self._parameters.pop(name, None)\n"
        "            self._modules.pop(name, None)\n"
        "        object.__setattr__(self, name, value)\n"
        "\n"
        "    def parameters(self):\n"
        "        for p in self._parameters.values():\n"
        "            yield p\n"
        "        for m in self._modules.values():\n"
        "            yield from m.parameters()\n"
        "\n"
        "    def train(self, mode: bool = True):\n"
        "        self.training = mode\n"
        "        for m in self._modules.values():\n"
        "            m.train(mode)\n"
        "        return self\n"
        "\n"
        "    def eval(self):\n"
        "        return self.train(False)\n"
        "\n"
        "    def forward(self, *args, **kwargs):\n"
        "        raise NotImplementedError()"
    ),
    solution_notes=(
        "**Why `eval` is a thin wrapper.** Keeping `eval()` as "
        "`self.train(False)` (rather than duplicating the recursion) means "
        "any future logic added to `train` — logging, hook firing, "
        "switching cuDNN deterministic flags — automatically applies to "
        "`eval`. PyTorch's actual `nn.Module.eval` is one line for this "
        "reason.\n\n"
        "**Bootstrap order is load-bearing.** Setting `training = True` in "
        "`__init__` MUST use `object.__setattr__`. The custom `__setattr__` "
        "would otherwise check `isinstance(value, Parameter)` / "
        "`isinstance(value, Module)` — fine here — but the BOOLEAN True "
        "falls to the else branch which calls `self._parameters.pop(name, "
        "None)`. If `_parameters` was already installed it's fine; if not, "
        "AttributeError. Always bootstrap with `object.__setattr__`."
    ),
)


# =========================================================================
# 4. parameter-wrap-around-tensor ex2: fix the anti-pattern by re-wrapping IS-A
# =========================================================================

SPEC_PARAMETER_FIX = _spec(
    atom_id="parameter-wrap-around-tensor",
    subtopic="Backprop: Parameter wrap around Tensor",
    topic_folder="prereqs_custom_tensor",
    needs_custom_tensor=True,
    recap=(
        "## Repairing the composition-Parameter anti-pattern — deepening\n"
        "\n"
        "Ex1 observed that `WrapParam(tensor)` (composition / HAS-A) is "
        "silently invisible to `isinstance(_, MiniTensor)` filters. The "
        "fix is to convert each WrapParam to an `IsAParam(MiniTensor)` "
        "subclass instance BEFORE the autograd layer ever sees it.\n"
        "\n"
        "```python\n"
        "def fix_params(things):\n"
        "    fixed = []\n"
        "    for p in things:\n"
        "        if isinstance(p, WrapParam):\n"
        "            fixed.append(IsAParam(p.tensor))\n"
        "        else:\n"
        "            fixed.append(p)\n"
        "    return fixed\n"
        "```\n"
        "\n"
        "**Why a conversion helper and not a class-rewrite.** In a real "
        "codebase you might inherit a config that uses the WrapParam form; "
        "you can't rewrite the class without breaking other consumers. A "
        "single 'normalize before training' pass is the safest fix.\n"
        "\n"
        "**Round-trip the fix.** After the conversion, every parameter "
        "passes `isinstance(_, MiniTensor)` — the autograd layer collects "
        "them all, the SGD step updates them all, the bug is gone."
    ),
    ex_title="fix WrapParam anti-pattern by converting to IsAParam in-place",
    slug="parameter-wrap-anti-pattern-fix-via-conversion",
    bloom="Apply",
    difficulty_num=3,
    keywords=["parameter", "fix", "conversion", "isinstance", "anti-pattern-repair"],
    kcs=["parameter-wrap-around-tensor", "parameter-subclass-of-tensor"],
    lo=(
        "Apply a conversion helper that maps `WrapParam` (composition) "
        "instances to `IsAParam` (subclass) instances while preserving "
        "tensor identity, then verify the round-trip survives the "
        "autograd-layer `isinstance` filter and a fake optimizer's update."
    ),
    prompt_body=(
        "Implement `fix_params(things)` and `fake_optimizer_step(params, "
        "lr)` together. The drill exercises the REPAIR side of the "
        "anti-pattern from ex1.\n\n"
        "Definitions you may reference (defined in the stub for you):\n"
        "- `WrapParam(tensor)` — composition Parameter. Stores `.tensor`. "
        "NOT a MiniTensor subclass.\n"
        "- `IsAParam(MiniTensor)` — subclass Parameter. Inherits.\n\n"
        "**1. `fix_params(things)` — list -> list.**\n"
        "Walk `things`. For each element:\n"
        "- If it's a `WrapParam`, build a new `IsAParam(p.tensor)` and "
        "include that.\n"
        "- Otherwise (already IsAParam, plain MiniTensor, or unrelated "
        "junk), include it unchanged.\n"
        "Return a NEW list. Don't mutate the input.\n\n"
        "**2. `fake_optimizer_step(params, lr)`.**\n"
        "Simulate what a real optimizer does to a parameter list:\n"
        "- Filter `params` with `isinstance(_, MiniTensor)`.\n"
        "- For each survivor, decrement `p.array` in place by "
        "`lr * t.ones_like(p.array)` (we don't have real gradients here — "
        "this stands in for a uniform 'step away from zero' move so the "
        "test can detect which params got updated).\n"
        "- Return the count of survivors (also = number of params that "
        "got updated).\n\n"
        "**Why fake_optimizer_step.** It's the minimal model of any "
        "autograd-layer helper. If `WrapParam`s are silently dropped, the "
        "count is wrong AND the `.tensor` stays unchanged. After "
        "`fix_params`, the count is right AND every `.array` shifts."
    ),
    stub=(
        "class WrapParam:\n"
        '    """Composition Parameter (anti-pattern from ex1)."""\n'
        "    def __init__(self, tensor):\n"
        "        self.tensor = tensor\n"
        "        self.requires_grad = True\n"
        "\n"
        "\n"
        "class IsAParam(MiniTensor):\n"
        '    """Subclass Parameter (correct design)."""\n'
        "    def __init__(self, array):\n"
        "        super().__init__(array, requires_grad=True)\n"
        "\n"
        "\n"
        "def fix_params(things: list) -> list:\n"
        '    """Convert WrapParams to IsAParams, leave others untouched."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def fake_optimizer_step(params: list, lr: float) -> int:\n"
        '    """Filter by isinstance(MiniTensor), apply uniform step, return count updated."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- invariant 1: WITHOUT the fix, fake_optimizer skips WrapParams ---\n"
        "t.manual_seed(0)\n"
        "raw_tensors = [t.tensor([1.0, 2.0]), t.tensor([3.0, 4.0]), t.tensor([5.0, 6.0])]\n"
        "mixed = [WrapParam(raw_tensors[0]), WrapParam(raw_tensors[1]), IsAParam(raw_tensors[2].clone())]\n"
        "n_before = fake_optimizer_step(mixed, lr=0.1)\n"
        "assert n_before == 1, (\n"
        "    f'WITHOUT fix: only the IsAParam should be updated; expected 1, got {n_before}'\n"
        ")\n"
        "# The two WrapParams' tensors are unchanged (the bug):\n"
        "assert t.allclose(mixed[0].tensor, t.tensor([1.0, 2.0])), 'WrapParam 0 silently skipped'\n"
        "assert t.allclose(mixed[1].tensor, t.tensor([3.0, 4.0])), 'WrapParam 1 silently skipped'\n"
        "\n"
        "# --- invariant 2: WITH the fix, all three get updated ---\n"
        "mixed2 = [WrapParam(t.tensor([1.0, 2.0])), WrapParam(t.tensor([3.0, 4.0])),\n"
        "          IsAParam(t.tensor([5.0, 6.0]))]\n"
        "fixed = fix_params(mixed2)\n"
        "assert len(fixed) == 3, f'fix_params must preserve list length, got {len(fixed)}'\n"
        "# Every element is now a MiniTensor instance.\n"
        "assert all(isinstance(p, MiniTensor) for p in fixed), (\n"
        "    f'every fixed element must pass isinstance(MiniTensor); got {[type(p).__name__ for p in fixed]}'\n"
        "    )\n"
        "n_after = fake_optimizer_step(fixed, lr=0.1)\n"
        "assert n_after == 3, f'after fix: all 3 must update; expected 3, got {n_after}'\n"
        "# Each .array is now shifted by lr (vector of ones), -> subtract 0.1\n"
        "for i, p in enumerate(fixed):\n"
        "    diff = p.array - t.tensor([1.0 + 2*i, 2.0 + 2*i])\n"
        "    # We subtracted 0.1 from each element, so diff == -0.1\n"
        "    assert t.allclose(diff, t.full_like(diff, -0.1), atol=1e-6), (\n"
        "        f'param {i} expected shift of -0.1 per element, got {diff}'\n"
        "    )\n"
        "\n"
        "# --- invariant 3: input list is not mutated ---\n"
        "src = [WrapParam(t.tensor([7.0])), IsAParam(t.tensor([8.0]))]\n"
        "snapshot_types = [type(p).__name__ for p in src]\n"
        "out = fix_params(src)\n"
        "assert [type(p).__name__ for p in src] == snapshot_types, (\n"
        "    f'fix_params must NOT mutate the input list: was {snapshot_types}, now {[type(p).__name__ for p in src]}'\n"
        ")\n"
        "assert out is not src, 'fix_params must return a new list, not the same one'\n"
        "\n"
        "# --- invariant 4: tensor identity preserved through conversion ---\n"
        "raw = t.tensor([42.0, 43.0])\n"
        "wp = WrapParam(raw)\n"
        "[fixed_one] = fix_params([wp])\n"
        "assert isinstance(fixed_one, IsAParam)\n"
        "assert fixed_one.array is raw, 'converting should reuse the underlying tensor, not copy'"
    ),
    solution_body=(
        "class WrapParam:\n"
        "    def __init__(self, tensor):\n"
        "        self.tensor = tensor\n"
        "        self.requires_grad = True\n"
        "\n"
        "\n"
        "class IsAParam(MiniTensor):\n"
        "    def __init__(self, array):\n"
        "        super().__init__(array, requires_grad=True)\n"
        "\n"
        "\n"
        "def fix_params(things: list) -> list:\n"
        "    out = []\n"
        "    for p in things:\n"
        "        if isinstance(p, WrapParam):\n"
        "            out.append(IsAParam(p.tensor))\n"
        "        else:\n"
        "            out.append(p)\n"
        "    return out\n"
        "\n"
        "\n"
        "def fake_optimizer_step(params: list, lr: float) -> int:\n"
        "    count = 0\n"
        "    for p in params:\n"
        "        if isinstance(p, MiniTensor):\n"
        "            p.array -= lr * t.ones_like(p.array)\n"
        "            count += 1\n"
        "    return count"
    ),
    solution_notes=(
        "**Reuse the underlying tensor on conversion.** "
        "`IsAParam(p.tensor)` hands the SAME `torch.Tensor` object into "
        "the new wrapper. Any external reference to the raw tensor still "
        "sees the in-place updates from the optimizer. Copying would "
        "duplicate memory AND break those references.\n\n"
        "**Why `fake_optimizer_step` returns the count.** It's the "
        "smallest observable that proves the filter worked. Without "
        "`fix_params`, count=1 (silent two-thirds drop). With it, count=3 "
        "(every param survived). Real bug-hunting in a codebase often "
        "starts with exactly this — print `len(list(model.parameters()))` "
        "and notice the number is wrong.\n\n"
        "**The lesson generalizes.** Anywhere `isinstance(_, Base)` "
        "gates behavior, COMPOSITION ('I hold a Base') invisibly fails "
        "while INHERITANCE ('I AM a Base') passes. The same pattern "
        "appears in framework hooks, plugin registries, and abstract "
        "type-class dispatch."
    ),
)


# =========================================================================
# 5. sgd-vanilla-from-scratch ex2: multi-step quadratic convergence + monotone loss
# =========================================================================

SPEC_SGD_CONVERGE = _spec(
    atom_id="sgd-vanilla-from-scratch",
    subtopic="Optimizer: SGD vanilla from scratch",
    topic_folder="prereqs_custom_tensor",
    needs_custom_tensor=True,
    recap=(
        "## SGD convergence on a quadratic — deepening\n"
        "\n"
        "Ex1 verified a single step. Ex2 verifies BEHAVIOR over many steps. "
        "On the convex quadratic `L(w) = 0.5 * (w - w*)^2`, the gradient "
        "is `dL/dw = w - w*` and the SGD recursion is:\n"
        "\n"
        "```\n"
        "w_{k+1} = w_k - lr * (w_k - w*) = (1 - lr) * w_k + lr * w*\n"
        "```\n"
        "\n"
        "For `0 < lr < 2`, the iterates contract toward `w*` at rate "
        "`|1 - lr|^k`. Two consequences:\n"
        "\n"
        "1. `w_k -> w*` (provided `lr < 2`).\n"
        "2. `L(w_k)` is **monotonically decreasing** in `k` (for "
        "`0 < lr <= 1`; for `lr > 1` it can oscillate but still converge "
        "if `lr < 2`).\n"
        "\n"
        "These two facts are how you sanity-check ANY new optimizer: run "
        "it on a quadratic, check (a) the optimum is reached, (b) the "
        "loss decreases each step at small lr."
    ),
    ex_title="run SGD on a quadratic; verify convergence + monotone loss decrease",
    slug="sgd-multi-step-quadratic-convergence",
    bloom="Apply",
    difficulty_num=3,
    keywords=["sgd", "convergence", "quadratic", "monotone-loss", "multi-step"],
    kcs=["sgd-vanilla-from-scratch", "convergence-on-quadratic"],
    lo=(
        "Apply vanilla SGD over many iterations to a 1-D quadratic loss, "
        "producing a sequence of weight + loss values that converge to "
        "the known minimum with monotonically decreasing loss."
    ),
    prompt_body=(
        "Implement `sgd_step(params, lr)` (same as ex1) and a driver "
        "`run_sgd_on_quadratic(w0, w_star, lr, n_steps)` that:\n\n"
        "1. Wrap `w0` (a Python float) as a MiniTensor with "
        "`requires_grad=True`.\n"
        "2. For `n_steps` iterations:\n"
        "   - Compute `loss = 0.5 * (w.array.item() - w_star) ** 2` (a "
        "Python float — we don't need the autograd graph for this drill).\n"
        "   - Compute `w.grad = w.array - w_star` (a tensor of shape "
        "matching `w.array`).\n"
        "   - Call `sgd_step([w], lr)`.\n"
        "   - Append the loss to a `losses` list AFTER the update is NOT "
        "needed — we append the loss as it was BEFORE the update, i.e. "
        "right after computing it.\n"
        "3. Return `(final_w_value, losses)` — `(float, list[float])`.\n\n"
        "**Logging order.** Record loss BEFORE the step so the list reads "
        "`[L(w_0), L(w_1), ..., L(w_{n-1})]`. The test verifies this list "
        "is monotonically non-increasing.\n\n"
        "**Don't use real autograd.** This drill is about the OPTIMIZER, "
        "not the graph. Set `w.grad` directly from the closed-form "
        "`w - w*`."
    ),
    stub=(
        "def sgd_step(params: list, lr: float) -> None:\n"
        '    """One SGD step (same as ex1)."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "\n"
        "def run_sgd_on_quadratic(w0: float, w_star: float, lr: float, n_steps: int):\n"
        '    """Return (final_w: float, losses: list[float])."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "# --- invariant 1: converges to w_star at moderate lr ---\n"
        "final, losses = run_sgd_on_quadratic(w0=0.0, w_star=5.0, lr=0.1, n_steps=200)\n"
        "assert isinstance(final, float), f'final must be float, got {type(final).__name__}'\n"
        "assert len(losses) == 200, f'losses must have n_steps entries, got {len(losses)}'\n"
        "assert abs(final - 5.0) < 1e-4, f'should converge to w_star=5, got {final}'\n"
        "\n"
        "# --- invariant 2: monotonically non-increasing loss at lr=0.1 (under 1) ---\n"
        "for i in range(len(losses) - 1):\n"
        "    assert losses[i + 1] <= losses[i] + 1e-9, (\n"
        "        f'loss must be non-increasing at lr=0.1; '\n"
        "        f'step {i}: {losses[i]} -> step {i+1}: {losses[i+1]}'\n"
        "    )\n"
        "\n"
        "# --- invariant 3: final loss is essentially zero ---\n"
        "assert losses[-1] < 1e-8, f'final loss should be ~0, got {losses[-1]}'\n"
        "\n"
        "# --- invariant 4: geometric decay rate matches theory ---\n"
        "# L(w_k) = 0.5 * |w_k - w*|^2 ; ratio of successive losses = (1-lr)^2 = 0.81 at lr=0.1\n"
        "# Compare losses[10] / losses[0]  ~  0.81^10 ~ 0.1216\n"
        "ratio = losses[10] / losses[0]\n"
        "expected_ratio = (1 - 0.1) ** (2 * 10)\n"
        "rel_err = abs(ratio - expected_ratio) / expected_ratio\n"
        "assert rel_err < 0.05, (\n"
        "    f'loss decay rate must match theory; got ratio {ratio:.5f} vs '\n"
        "    f'expected {expected_ratio:.5f} (rel err {rel_err:.4f})'\n"
        ")\n"
        "\n"
        "# --- invariant 5: starting at the minimum -> zero loss, zero motion ---\n"
        "final_at_min, losses_at_min = run_sgd_on_quadratic(w0=5.0, w_star=5.0, lr=0.1, n_steps=20)\n"
        "assert abs(final_at_min - 5.0) < 1e-9, 'starting at minimum: w should not move'\n"
        "assert all(L < 1e-12 for L in losses_at_min), 'starting at minimum: every loss must be zero'"
    ),
    solution_body=(
        "def sgd_step(params: list, lr: float) -> None:\n"
        "    for p in params:\n"
        "        if p.grad is None:\n"
        "            continue\n"
        "        p.array -= lr * p.grad\n"
        "        p.grad = None\n"
        "\n"
        "\n"
        "def run_sgd_on_quadratic(w0: float, w_star: float, lr: float, n_steps: int):\n"
        "    w = MiniTensor(t.tensor([w0]), requires_grad=True)\n"
        "    losses = []\n"
        "    for _ in range(n_steps):\n"
        "        # record loss before the step\n"
        "        loss = 0.5 * (w.array.item() - w_star) ** 2\n"
        "        losses.append(loss)\n"
        "        # closed-form grad of 0.5 * (w - w*)^2 is (w - w*)\n"
        "        w.grad = w.array - w_star\n"
        "        sgd_step([w], lr)\n"
        "    return float(w.array.item()), losses"
    ),
    solution_notes=(
        "**Geometric convergence is the fingerprint of SGD on a "
        "quadratic.** The recursion `w_{k+1} = (1 - lr) * w_k + lr * w*` "
        "is a 1-D linear contraction with rate `|1 - lr|`. Loss decays as "
        "`(1 - lr)^{2k}`, which is what test invariant 4 asserts. If you "
        "see a different rate in your implementation — likely off by a "
        "factor of `lr` somewhere — this is the test that catches it.\n\n"
        "**`losses` records BEFORE the step.** So `losses[0]` is the "
        "initial loss (largest), `losses[-1]` is the loss at the "
        "second-to-last iterate. After the final step, `final_w` itself "
        "has converged but we don't append a final loss. (Pick a "
        "convention and stick to it; this matches PyTorch's typical "
        "training-loop pattern.)\n\n"
        "**Don't use autograd here.** The atom under study is the "
        "OPTIMIZER. Wiring `w.grad` directly to the closed-form derivative "
        "isolates the test from any autograd bugs."
    ),
)


# =========================================================================
# 6. bce-log-loss-real-fake ex2: stable logits form via BCE-with-logits
# =========================================================================

SPEC_BCE_LOGITS = _spec(
    atom_id="bce-log-loss-real-fake",
    subtopic="GAN: BCE log loss real/fake",
    topic_folder="prereqs_dcgan_final",
    needs_custom_tensor=False,
    recap=(
        "## BCE-with-logits — numerically stable D loss — deepening\n"
        "\n"
        "Ex1 used `F.binary_cross_entropy(probs, targets)` — the version "
        "that expects post-sigmoid probabilities. In production GAN code "
        "you should feed LOGITS to `F.binary_cross_entropy_with_logits` "
        "instead — it fuses `sigmoid + bce` internally with the "
        "log-sum-exp trick:\n"
        "\n"
        "```python\n"
        "loss_real = F.binary_cross_entropy_with_logits(D_logits_real, t.ones_like(D_logits_real))\n"
        "loss_fake = F.binary_cross_entropy_with_logits(D_logits_fake, t.zeros_like(D_logits_fake))\n"
        "loss_D = loss_real + loss_fake\n"
        "```\n"
        "\n"
        "**Why this matters.** When D becomes very confident on a batch "
        "(logits at ±20), the sigmoid output is `1 - 1e-9` or `1e-9`, and "
        "`log(1 - p)` or `log(p)` plus float32 rounding crash to `inf` "
        "(or `nan`) in the bare `binary_cross_entropy` path. The fused "
        "form sidesteps that — see `softplus(x) = log(1 + exp(x))` for the "
        "exact algebra.\n"
        "\n"
        "**Numerical equivalence.** For non-extreme logits (in `[-10, "
        "+10]`-ish), the logits form and the probs form agree to "
        "single-precision."
    ),
    ex_title="discriminator loss via BCE-with-logits (numerically stable form)",
    slug="discriminator-bce-with-logits-stable",
    bloom="Apply",
    difficulty_num=3,
    keywords=["bce-with-logits", "numerical-stability", "gan", "logits-form"],
    kcs=["bce-log-loss-real-fake", "bce-with-logits-fused"],
    lo=(
        "Apply `F.binary_cross_entropy_with_logits` with ones/zeros "
        "targets to compute the discriminator loss directly from logits, "
        "and verify numerical equivalence to the probs form at moderate "
        "logits plus stability advantage at extreme logits."
    ),
    prompt_body=(
        "Implement `ex2_discriminator_loss_logits(d_logits_real, "
        "d_logits_fake)`. The numerically-stable form of the discriminator "
        "loss:\n\n"
        "1. `d_logits_real` are D's RAW LOGIT outputs on real images "
        "(shape `(B,)`, ANY real-valued float — pre-sigmoid).\n"
        "2. `d_logits_fake` are D's raw logits on fakes (same shape, same "
        "values range).\n"
        "3. Build target tensors:\n"
        "   - `real_t = t.ones_like(d_logits_real)`\n"
        "   - `fake_t = t.zeros_like(d_logits_fake)`\n"
        "4. Compute `loss_real = F.binary_cross_entropy_with_logits(d_logits_real, real_t)`.\n"
        "5. Compute `loss_fake = F.binary_cross_entropy_with_logits(d_logits_fake, fake_t)`.\n"
        "6. Return `loss_real + loss_fake` — a scalar tensor.\n\n"
        "**Do NOT call sigmoid then `F.binary_cross_entropy`.** The whole "
        "point is to keep the operation in logit-space so the fused kernel "
        "uses `softplus` instead of `log(sigmoid(x))`. The test stresses "
        "this with logits at ±50 where the unfused form is `inf`/`nan`.\n\n"
        "Returns a scalar tensor."
    ),
    stub=(
        "def ex2_discriminator_loss_logits(d_logits_real: Tensor, d_logits_fake: Tensor) -> Tensor:\n"
        '    """BCE-with-logits(D_real, 1) + BCE-with-logits(D_fake, 0)."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "import torch.nn.functional as F\n"
        "import math\n"
        "\n"
        "# --- invariant 1: matches probs form on moderate logits ---\n"
        "t.manual_seed(0)\n"
        "logits_real = t.randn(16) * 2.0     # in roughly (-6, 6)\n"
        "logits_fake = t.randn(16) * 2.0\n"
        "ours = ex2_discriminator_loss_logits(logits_real, logits_fake)\n"
        "# Compare to the bare-probs form on the same data\n"
        "probs_real = t.sigmoid(logits_real)\n"
        "probs_fake = t.sigmoid(logits_fake)\n"
        "ref = F.binary_cross_entropy(probs_real, t.ones_like(probs_real)) + \\\n"
        "      F.binary_cross_entropy(probs_fake, t.zeros_like(probs_fake))\n"
        "assert t.allclose(ours, ref, atol=1e-5), (\n"
        "    f'must match probs form at moderate logits; ours={ours.item()}, ref={ref.item()}'\n"
        ")\n"
        "\n"
        "# --- invariant 2: STABLE at extreme logits (±50) — does not overflow ---\n"
        "extreme_real = t.tensor([50.0, 50.0, 50.0, 50.0])     # D is overwhelmingly sure 'real'\n"
        "extreme_fake = t.tensor([-50.0, -50.0, -50.0, -50.0]) # D is overwhelmingly sure 'fake'\n"
        "loss_stable = ex2_discriminator_loss_logits(extreme_real, extreme_fake)\n"
        "assert t.isfinite(loss_stable).item(), (\n"
        "    f'extreme logits must NOT produce inf/nan with BCE-with-logits; got {loss_stable.item()}'\n"
        ")\n"
        "# Loss should be very small (near zero — D is correct on every example).\n"
        "assert loss_stable.item() < 1e-15, (\n"
        "    f'D correct at extreme logits should give ~0 loss, got {loss_stable.item()}'\n"
        ")\n"
        "\n"
        "# --- invariant 3: D MAXIMALLY WRONG at extreme logits → finite, large loss ---\n"
        "wrong_real = t.tensor([-50.0, -50.0])   # D says 'fake' for reals\n"
        "wrong_fake = t.tensor([50.0, 50.0])     # D says 'real' for fakes\n"
        "loss_wrong = ex2_discriminator_loss_logits(wrong_real, wrong_fake)\n"
        "assert t.isfinite(loss_wrong).item(), f'must be finite, got {loss_wrong.item()}'\n"
        "# For extreme wrong logits, per-sample loss ~= |logit|, summed over 2 batches of 2 -> ~100\n"
        "assert loss_wrong.item() > 50.0, f'wrong-extreme loss should be large, got {loss_wrong.item()}'\n"
        "\n"
        "# --- invariant 4: matches torch's bce-with-logits witness exactly ---\n"
        "rnd_r = t.randn(8)\n"
        "rnd_f = t.randn(8)\n"
        "got = ex2_discriminator_loss_logits(rnd_r, rnd_f)\n"
        "ref2 = F.binary_cross_entropy_with_logits(rnd_r, t.ones_like(rnd_r)) + \\\n"
        "       F.binary_cross_entropy_with_logits(rnd_f, t.zeros_like(rnd_f))\n"
        "assert t.allclose(got, ref2, atol=1e-6), f'must match F.binary_cross_entropy_with_logits exactly'\n"
        "\n"
        "# --- invariant 5: zero logits -> 2 * log(2) (coin-flip case) ---\n"
        "zero_r = t.zeros(8)\n"
        "zero_f = t.zeros(8)\n"
        "loss_coin = ex2_discriminator_loss_logits(zero_r, zero_f)\n"
        "expected_coin = 2 * math.log(2)\n"
        "assert abs(loss_coin.item() - expected_coin) < 1e-5, (\n"
        "    f'logits=0 → coin-flip loss 2*log(2) = {expected_coin:.4f}, got {loss_coin.item()}'\n"
        ")"
    ),
    solution_body=(
        "def ex2_discriminator_loss_logits(d_logits_real: Tensor, d_logits_fake: Tensor) -> Tensor:\n"
        "    import torch.nn.functional as F\n"
        "    loss_real = F.binary_cross_entropy_with_logits(d_logits_real, t.ones_like(d_logits_real))\n"
        "    loss_fake = F.binary_cross_entropy_with_logits(d_logits_fake, t.zeros_like(d_logits_fake))\n"
        "    return loss_real + loss_fake"
    ),
    solution_notes=(
        "**Why the fused form is stable.** "
        "`bce_with_logits(x, 1) = -log(sigmoid(x)) = softplus(-x)`, and "
        "`softplus` is implemented as `log1p(exp(-|x|)) + max(x, 0)` — "
        "the absolute-value trick guarantees `exp` always sees a "
        "non-positive argument. So `softplus(50) ≈ 50` (computed exactly) "
        "while `-log(sigmoid(50)) = -log(1 - 1e-22)` underflows to `inf` "
        "in float32.\n\n"
        "**When to use which.** Always prefer `binary_cross_entropy_with_"
        "logits` in production. The plain `binary_cross_entropy` is "
        "occasionally useful for plotting losses against PREDICTED "
        "probabilities (interpretability), but never inside the training "
        "loop.\n\n"
        "**Gradient is also stable.** "
        "`d/dx bce_with_logits(x, target) = sigmoid(x) - target`, "
        "computed without ever materializing `1 - sigmoid(x)` near 1."
    ),
    extra_imports=["import matplotlib.pyplot as plt"],
)


# =========================================================================
# 7. bn-weight-bias-init-pattern ex2: near-identity at init
# =========================================================================

SPEC_BN_IDENTITY = _spec(
    atom_id="bn-weight-bias-init-pattern",
    subtopic="GAN: BN weight=1 bias=0 init",
    topic_folder="prereqs_dcgan_final",
    needs_custom_tensor=False,
    recap=(
        "## BN at init is a near-identity transform — deepening\n"
        "\n"
        "Ex1 verified `weight ~ N(1, 0.02)` and `bias = 0` per-element. "
        "The deeper question: WHY those values? Answer: because the "
        "affine map then approximates the IDENTITY on already-normalized "
        "features.\n"
        "\n"
        "BatchNorm in `.eval()` mode applies\n"
        "```\n"
        "y = (x - running_mean) / sqrt(running_var + eps) * weight + bias\n"
        "```\n"
        "\n"
        "If `running_mean ≈ 0`, `running_var ≈ 1` (the default), "
        "`weight ≈ 1`, `bias = 0`, then `y ≈ x`. The BN layer at init is "
        "essentially a pass-through, so a freshly-initialized DCGAN doesn't "
        "have wild scale/shift surprises in its forward pass.\n"
        "\n"
        "**`std=0.02` is a tiny perturbation.** Big enough to break "
        "channel-wise symmetry. Small enough not to disturb the "
        "near-identity property."
    ),
    ex_title="verify BN at init is near-identity on already-normalized inputs",
    slug="bn-init-near-identity-property",
    bloom="Analyze",
    difficulty_num=3,
    keywords=["batchnorm", "init", "identity-at-init", "scale-shift", "near-identity"],
    kcs=["bn-weight-bias-init-pattern", "bn-affine-near-identity"],
    lo=(
        "Analyze the BatchNorm-at-init contract by applying the DCGAN init "
        "to a fresh BN layer, feeding already-normalized inputs in eval "
        "mode, and verifying the output deviates from input by at most "
        "the std=0.02 init noise."
    ),
    prompt_body=(
        "Implement `ex2_apply_bn_init_and_check(channels, batch_size, "
        "n_spatial)` that builds a single `nn.BatchNorm2d(channels)` "
        "layer, applies the DCGAN init, runs an already-normalized input "
        "through it in eval mode, and returns a triple `(layer, x, y)`:\n\n"
        "1. Build `layer = nn.BatchNorm2d(channels)`.\n"
        "2. Apply the DCGAN BN init INLINE — directly on `layer` "
        "(don't call a previous-ex function):\n"
        "   - `nn.init.normal_(layer.weight, 1.0, 0.02)`\n"
        "   - `nn.init.zeros_(layer.bias)`\n"
        "3. Build a fixed-shape standard-normal input "
        "`x = t.randn(batch_size, channels, n_spatial, n_spatial)`.\n"
        "4. Switch the layer to eval mode (`layer.eval()`) so it uses "
        "`running_mean=0, running_var=1` (the defaults at init).\n"
        "5. Run `y = layer(x)` under `t.no_grad()`.\n"
        "6. Return `(layer, x, y)`.\n\n"
        "The test cell then verifies that `y` is element-wise close to "
        "`x * gamma + 0` where `gamma` is the per-channel BN weight — "
        "i.e. the BN is exactly the affine map at init, no normalization "
        "is being applied because the running stats are the identity."
    ),
    stub=(
        "def ex2_apply_bn_init_and_check(channels: int, batch_size: int, n_spatial: int):\n"
        '    """Build BN, apply DCGAN init, return (layer, x, y) at eval-mode forward."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "import torch.nn as nn\n"
        "\n"
        "# --- invariant 1: returned tuple shape + types ---\n"
        "layer, x, y = ex2_apply_bn_init_and_check(channels=8, batch_size=4, n_spatial=5)\n"
        "assert isinstance(layer, nn.BatchNorm2d), f'expected BatchNorm2d, got {type(layer).__name__}'\n"
        "assert x.shape == (4, 8, 5, 5), f'x shape: {x.shape}'\n"
        "assert y.shape == x.shape, f'y shape mismatch: {y.shape}'\n"
        "\n"
        "# --- invariant 2: gamma is near 1, beta is exactly 0 ---\n"
        "assert abs(layer.weight.mean().item() - 1.0) < 0.05, (\n"
        "    f'gamma mean ~1 expected, got {layer.weight.mean().item():.4f}'\n"
        ")\n"
        "assert layer.weight.std().item() < 0.05, (\n"
        "    f'gamma std should be ~0.02, got {layer.weight.std().item():.4f}'\n"
        ")\n"
        "assert t.all(layer.bias == 0), f'beta must be exactly zero, got mean {layer.bias.mean().item()}'\n"
        "\n"
        "# --- invariant 3: near-identity property — y ≈ x * gamma (per channel) ---\n"
        "# At init: running_mean=0, running_var=1, beta=0, so y = (x - 0) / sqrt(1 + eps) * gamma + 0\n"
        "#       ≈ x * gamma (since eps is ~1e-5)\n"
        "gamma = layer.weight.detach().view(1, -1, 1, 1)   # broadcast over batch + spatial\n"
        "expected = x * gamma\n"
        "max_diff = (y - expected).abs().max().item()\n"
        "# eps in BN is 1e-5; sqrt(1 + 1e-5) ~ 1 + 5e-6, so the rescaling factor is essentially 1.\n"
        "assert max_diff < 1e-3, (\n"
        "    f'BN at init must be near-identity y = x * gamma; max diff = {max_diff:.5f}'\n"
        ")\n"
        "\n"
        "# --- invariant 4: deviation from PURE identity is bounded by gamma std ---\n"
        "# y / x ≈ gamma. If gamma were exactly 1, y == x. Since gamma ~ N(1, 0.02), the\n"
        "# elementwise ratio scatter is ~0.02.\n"
        "# Approximate ratio std:\n"
        "# Use a fresh normalized input (avoid division-by-near-zero artifacts).\n"
        "x_clean = t.full((1, 8, 1, 1), 1.0)   # constant input, makes ratio = gamma exactly\n"
        "layer.eval()\n"
        "with t.no_grad():\n"
        "    y_clean = layer(x_clean)\n"
        "ratio = (y_clean / x_clean).flatten()\n"
        "assert ratio.std().item() < 0.05, (\n"
        "    f'y/x scatter at init should be bounded by gamma std (~0.02), got {ratio.std().item():.4f}'\n"
        ")\n"
        "assert abs(ratio.mean().item() - 1.0) < 0.05, (\n"
        "    f'mean ratio should be ~1 (identity-on-average), got {ratio.mean().item():.4f}'\n"
        ")\n"
        "\n"
        "# --- invariant 5: with larger channels, near-identity property holds ---\n"
        "layer2, x2, y2 = ex2_apply_bn_init_and_check(channels=64, batch_size=8, n_spatial=4)\n"
        "g2 = layer2.weight.detach().view(1, -1, 1, 1)\n"
        "max_diff_big = (y2 - x2 * g2).abs().max().item()\n"
        "assert max_diff_big < 1e-3, f'near-identity must hold at C=64, got max diff {max_diff_big}'"
    ),
    solution_body=(
        "def ex2_apply_bn_init_and_check(channels: int, batch_size: int, n_spatial: int):\n"
        "    import torch.nn as nn\n"
        "    layer = nn.BatchNorm2d(channels)\n"
        "    nn.init.normal_(layer.weight, 1.0, 0.02)\n"
        "    nn.init.zeros_(layer.bias)\n"
        "    x = t.randn(batch_size, channels, n_spatial, n_spatial)\n"
        "    layer.eval()\n"
        "    with t.no_grad():\n"
        "        y = layer(x)\n"
        "    return layer, x, y"
    ),
    solution_notes=(
        "**Why the eval-mode forward is essential to the test.** In "
        "train mode, BN computes mean/var on the input batch itself (and "
        "the result of `y = (x - batch_mean) / batch_std * gamma + beta` "
        "is NOT `x * gamma` — it's `(normalized(x)) * gamma`, which "
        "destroys the original signal). Eval mode uses the running stats "
        "(default 0, 1 at init), making BN a pure affine map.\n\n"
        "**Why DCGAN doesn't simply use `nn.init.ones_(layer.weight)`.** "
        "Setting every channel's gamma to exactly 1 would make the BN "
        "layer perfectly symmetric — every channel has the same scale. "
        "The `std=0.02` jitter breaks that symmetry so each channel can "
        "learn a slightly different scale during training.\n\n"
        "**Numerical bound is `eps`-driven.** `BatchNorm2d.eps` defaults "
        "to `1e-5`, so the rescaling factor `1 / sqrt(running_var + eps)` "
        "= `1 / sqrt(1.00001) ≈ 0.999995`. The drift from pure-identity "
        "is at the eps-level — well under 1e-3 across all entries, "
        "matching invariant 3."
    ),
    extra_imports=["import torch.nn as nn", "import matplotlib.pyplot as plt"],
)


# =========================================================================
# 8. channel-list-reverse-build ex2: assemble decoder Sequential + verify doubling
# =========================================================================

SPEC_CHANNEL_DECODER = _spec(
    atom_id="channel-list-reverse-build",
    subtopic="GAN: channel-list reverse build",
    topic_folder="prereqs_dcgan_final",
    needs_custom_tensor=False,
    recap=(
        "## Assembling a decoder from channel pairs — deepening\n"
        "\n"
        "Ex1 produced a list of `(in_c, out_c)` pairs for the decoder. "
        "Ex2 takes that list and builds an actual "
        "`nn.Sequential(ConvTranspose2d, BN, ReLU, ...)` decoder that "
        "doubles spatial resolution per block:\n"
        "\n"
        "```python\n"
        "blocks = []\n"
        "for in_c, out_c in decoder_pairs:\n"
        "    blocks.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False))\n"
        "    blocks.append(nn.BatchNorm2d(out_c))\n"
        "    blocks.append(nn.ReLU(inplace=True))\n"
        "decoder = nn.Sequential(*blocks)\n"
        "```\n"
        "\n"
        "**`stride=2 + kernel=4 + padding=1` doubles spatial dims.** "
        "Output size for ConvTranspose2d is "
        "`(H_in - 1) * stride - 2 * padding + kernel_size` "
        "= `(H_in - 1) * 2 - 2 + 4 = 2 * H_in`. The drill verifies "
        "this with a tiny input.\n"
        "\n"
        "**No activation on the FINAL block.** A real DCGAN ends in "
        "Tanh — but to keep this drill focused on the assembly mechanic, "
        "every block (including the last) gets the BN + ReLU triplet. "
        "The test only checks shapes, not activation choice."
    ),
    ex_title="assemble decoder Sequential from channel pairs; verify spatial doubling",
    slug="channel-pair-decoder-sequential-build",
    bloom="Apply",
    difficulty_num=3,
    keywords=["decoder", "sequential", "convtranspose", "spatial-doubling", "channel-pairs"],
    kcs=["channel-list-reverse-build", "convtranspose-double-spatial"],
    lo=(
        "Apply consecutive-pair iteration over a reversed channel list "
        "to build an `nn.Sequential` decoder of ConvTranspose2d+BN+ReLU "
        "blocks whose forward doubles spatial resolution per block."
    ),
    prompt_body=(
        "Implement `ex2_build_decoder(encoder_channels)`. Given an "
        "encoder channel list (e.g. `[3, 64, 128, 256, 512]`), build the "
        "MIRROR decoder as a single `nn.Sequential`:\n\n"
        "1. Build `decoder_channels = encoder_channels[::-1]` (slice "
        "reverse). For `[3, 64, 128, 256, 512]` -> `[512, 256, 128, 64, 3]`.\n"
        "2. Build `decoder_pairs = list(zip(decoder_channels[:-1], "
        "decoder_channels[1:]))`. For the example: "
        "`[(512, 256), (256, 128), (128, 64), (64, 3)]`.\n"
        "3. For each `(in_c, out_c)` in `decoder_pairs`, append THREE "
        "modules to a `blocks` list:\n"
        "   - `nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, "
        "padding=1, bias=False)`\n"
        "   - `nn.BatchNorm2d(out_c)`\n"
        "   - `nn.ReLU(inplace=True)`\n"
        "4. Return `nn.Sequential(*blocks)`.\n\n"
        "**Why `bias=False` on the ConvTranspose.** BatchNorm immediately "
        "follows and re-learns the per-channel bias as its beta. The "
        "ConvTranspose bias would be redundant + slightly wasteful.\n\n"
        "**Use `inplace=True` on ReLU** to save memory (a DCGAN "
        "convention — saves ~10% on the activation buffer).\n\n"
        "The test cell builds the decoder for a known encoder list and "
        "verifies (a) the layer counts are right, (b) the spatial dims "
        "exactly double per block."
    ),
    stub=(
        "def ex2_build_decoder(encoder_channels: list[int]) -> nn.Module:\n"
        '    """Build a Sequential decoder of ConvT+BN+ReLU blocks from reversed channels."""\n'
        "    raise NotImplementedError()"
    ),
    test_body=(
        "import torch.nn as nn\n"
        "\n"
        "# --- invariant 1: returns Sequential with the right module count ---\n"
        "encoder = [3, 64, 128, 256, 512]\n"
        "decoder = ex2_build_decoder(encoder)\n"
        "assert isinstance(decoder, nn.Sequential), f'expected Sequential, got {type(decoder).__name__}'\n"
        "# 4 decoder_pairs -> 4 blocks * 3 modules = 12 modules total\n"
        "n_modules = len(list(decoder))\n"
        "assert n_modules == 12, f'expected 12 modules (4 blocks * ConvT+BN+ReLU), got {n_modules}'\n"
        "\n"
        "# --- invariant 2: channel pattern of the ConvT layers is the reversed list ---\n"
        "convt_layers = [m for m in decoder if isinstance(m, nn.ConvTranspose2d)]\n"
        "assert len(convt_layers) == 4, f'expected 4 ConvTranspose2d, got {len(convt_layers)}'\n"
        "expected_pairs = [(512, 256), (256, 128), (128, 64), (64, 3)]\n"
        "for i, (in_c, out_c) in enumerate(expected_pairs):\n"
        "    layer = convt_layers[i]\n"
        "    assert layer.in_channels == in_c and layer.out_channels == out_c, (\n"
        "        f'ConvT {i}: expected ({in_c}, {out_c}), got ({layer.in_channels}, {layer.out_channels})'\n"
        "    )\n"
        "    assert layer.bias is None, f'ConvT {i} must have bias=False (bias is None)'\n"
        "\n"
        "# --- invariant 3: spatial dim doubles per block (forward pass at H=4) ---\n"
        "x = t.zeros(1, 512, 4, 4)\n"
        "decoder.eval()\n"
        "with t.no_grad():\n"
        "    y = decoder(x)\n"
        "# 4 blocks of stride-2 ConvT -> H,W go 4 -> 8 -> 16 -> 32 -> 64\n"
        "assert y.shape == (1, 3, 64, 64), f'final shape (4 doublings of 4): {y.shape}'\n"
        "\n"
        "# --- invariant 4: per-block doubling — probe between blocks ---\n"
        "# Extract progressive output by feeding through prefixes of the decoder.\n"
        "expected_h = [8, 16, 32, 64]\n"
        "x2 = t.zeros(1, 512, 4, 4)\n"
        "decoder.eval()\n"
        "with t.no_grad():\n"
        "    cur = x2\n"
        "    for block_idx in range(4):\n"
        "        # 3 modules per block: ConvT, BN, ReLU\n"
        "        for sub_idx in range(3):\n"
        "            cur = decoder[block_idx * 3 + sub_idx](cur)\n"
        "        assert cur.shape[-1] == expected_h[block_idx], (\n"
        "            f'after block {block_idx}, expected H={expected_h[block_idx]}, got {cur.shape[-1]}'\n"
        "        )\n"
        "\n"
        "# --- invariant 5: different channel lists work too ---\n"
        "small = [1, 16, 64]\n"
        "dec_small = ex2_build_decoder(small)\n"
        "# 2 pairs * 3 modules each\n"
        "assert len(list(dec_small)) == 6, f'small decoder: expected 6 modules, got {len(list(dec_small))}'\n"
        "convt_small = [m for m in dec_small if isinstance(m, nn.ConvTranspose2d)]\n"
        "assert (convt_small[0].in_channels, convt_small[0].out_channels) == (64, 16)\n"
        "assert (convt_small[1].in_channels, convt_small[1].out_channels) == (16, 1)\n"
        "# Spatial: 4 -> 8 -> 16\n"
        "x_small = t.zeros(1, 64, 4, 4)\n"
        "dec_small.eval()\n"
        "with t.no_grad():\n"
        "    y_small = dec_small(x_small)\n"
        "assert y_small.shape == (1, 1, 16, 16), f'small decoder out: {y_small.shape}'"
    ),
    solution_body=(
        "def ex2_build_decoder(encoder_channels):\n"
        "    import torch.nn as nn\n"
        "    decoder_channels = encoder_channels[::-1]\n"
        "    decoder_pairs = list(zip(decoder_channels[:-1], decoder_channels[1:]))\n"
        "    blocks = []\n"
        "    for in_c, out_c in decoder_pairs:\n"
        "        blocks.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False))\n"
        "        blocks.append(nn.BatchNorm2d(out_c))\n"
        "        blocks.append(nn.ReLU(inplace=True))\n"
        "    return nn.Sequential(*blocks)"
    ),
    solution_notes=(
        "**Why stride=2, kernel=4, padding=1 specifically.** This is "
        "the DCGAN paper's recipe for exact 2x spatial expansion. The "
        "general ConvTranspose output formula is "
        "`(H_in - 1) * stride - 2 * padding + kernel_size + output_padding`. "
        "Plugging in: `(H-1) * 2 - 2 + 4 = 2 * H`. Other kernel/stride "
        "combos can also double — e.g. `kernel=2, stride=2, padding=0` — "
        "but the kernel=4 form has more receptive-field overlap, which "
        "reduces checkerboard artifacts in generated images.\n\n"
        "**`bias=False` + BatchNorm is the universal pattern.** The BN "
        "layer's `beta` parameter subsumes whatever bias the conv would "
        "have provided. Keeping conv bias means redundant parameters and "
        "a small numerical asymmetry — measurable in benchmarks, "
        "uniformly avoided in practice.\n\n"
        "**`inplace=True` on ReLU saves activation memory.** A 64x64 "
        "feature map at float32 is 16KB per channel — across 4 blocks of "
        "a ResNet-like net, the saved buffer can be a meaningful fraction "
        "of GPU RAM. The trade-off: in-place ops can fool autograd in "
        "unusual graph patterns, but plain ReLU is safe."
    ),
    extra_imports=["import torch.nn as nn", "import matplotlib.pyplot as plt"],
)


# ---------------------------------------------------------------- assembly

ALL_SPECS = [
    SPEC_LINEAR_BIAS_BACK,
    SPEC_LOGSUMEXP_GRAD,
    SPEC_MODULE_TRAIN_EVAL,
    SPEC_PARAMETER_FIX,
    SPEC_SGD_CONVERGE,
    SPEC_BCE_LOGITS,
    SPEC_BN_IDENTITY,
    SPEC_CHANNEL_DECODER,
]


def _verify_all(specs):
    import torch as t
    import numpy as np
    from torch import Tensor

    passed = 0
    failed = []

    for spec in specs:
        ex_id = f"ex{spec['exercise_index']}"
        tag = f"{spec['atom_id']}/{ex_id}"
        ns = {
            "t": t,
            "np": np,
            "Tensor": Tensor,
            "_dd_passed": set(),
            "__name__": "__main__",
        }
        t.manual_seed(0)
        np.random.seed(0)

        # If the spec needs the custom-tensor preamble, exec it first.
        if any(_CUSTOM_TENSOR_PREAMBLE in s for s in spec.get("extra_imports") or []):
            try:
                exec(_CUSTOM_TENSOR_PREAMBLE, ns)
            except Exception as e:
                failed.append((tag, repr(e), traceback.format_exc()))
                continue

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
    print(f"[deepening_p_batch10] Verifying {len(ALL_SPECS)} specs against torch backend...")
    _verify_all(ALL_SPECS)

    print(f"\n[deepening_p_batch10] All verified — emitting notebooks.")
    for spec in ALL_SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_p_batch10] {len(ALL_SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
