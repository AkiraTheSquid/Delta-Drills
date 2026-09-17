#!/usr/bin/env python3
"""Author 8 ex2 deepening drills (batch 12, group Z).

Atoms (4 resnet-modules + 4 tensor-utils):
    - replace-final-head        (ex2: freeze backbone, verify grad only on head)
    - residual-skip-add         (ex2: identity vs 1x1-conv shortcut shape contrast)
    - resnet-stem               (ex2: no-stem alt vs stem — output shape contrast)
    - state-dict-load           (ex2: strict=True raises MissingKey; catch + report)
    - cuda-empty-cache          (ex2: CPU no-op path via is_available() guard)
    - detach-clone-snapshot     (ex2: .detach() shares storage vs .detach().clone())
    - index-by-tensor           (ex2: advanced indexing with TWO index tensors)
    - matvec                    (ex2: batched matvec — bmm vs einsum bij,bj->bi)

Each ex2 hits a DISTINCT facet from ex1. ONE LO + ONE Bloom + <=2 KCs per drill.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC_RESNET = "prereqs_resnet_modules"
TOPIC_TENSOR = "prereqs_tensor_utils"


# ---------------------------------------------------------------------------
# Recap blocks
# ---------------------------------------------------------------------------

RECAP_FREEZE_HEAD = (
    "## Freeze the backbone, train only the new head\n"
    "\n"
    "Ex1 SWAPPED the final classifier. The deepening move is what almost "
    "always comes next in transfer learning: freeze the backbone so the "
    "optimizer only updates the new head's parameters.\n"
    "\n"
    "```python\n"
    "model.fc = nn.Linear(in_features, new_num_classes)\n"
    "for p in model.backbone.parameters():\n"
    "    p.requires_grad_(False)\n"
    "```\n"
    "\n"
    "**`requires_grad_(False)` is in-place.** The trailing underscore "
    "mutates the tensor's flag without returning a new one. Setting "
    "`p.requires_grad = False` directly also works, but the trailing-"
    "underscore form is the canonical PyTorch idiom for in-place flag "
    "updates.\n"
    "\n"
    "**Why the new head is automatically trainable.** Brand-new "
    "`nn.Linear(...)` parameters default to `requires_grad=True`. You only "
    "have to flip the OLD layers off — the new layer is already on.\n"
    "\n"
    "**Verifying the freeze worked.** After a `loss.backward()`, only the "
    "head params should have non-`None` `.grad`. The backbone params have "
    "`.grad is None` because no graph was built for them."
)

RECAP_SKIP_SHAPE_CONTRAST = (
    "## Identity vs 1×1-conv shortcut — what each branch produces\n"
    "\n"
    "Ex1 BUILT the conditional ResidualBlock. The deepening move is to "
    "INSPECT both branches on a shape-mismatched case — show how the "
    "1×1 shortcut rescues the add when channels (or stride) differ.\n"
    "\n"
    "Two scenarios:\n"
    "\n"
    "- **Matched (in=out, stride=1):** `self.skip = nn.Identity()`. "
    "Identity returns its input unchanged. `conv(x) + skip(x)` adds the "
    "fresh conv output to the ORIGINAL input directly. Zero extra params, "
    "zero FLOPs from the skip branch.\n"
    "- **Mismatched (in≠out OR stride>1):** `self.skip = nn.Conv2d(in, "
    "out, kernel_size=1, stride=first_stride)`. The 1×1 conv changes "
    "channel count (and downsamples spatially via stride) without "
    "introducing receptive-field artifacts — every output pixel is a "
    "linear combo of the SAME input pixel across input channels.\n"
    "\n"
    "**Why a 1×1 conv, not a Linear.** A spatial tensor `(B, C, H, W)` "
    "needs the same `H, W` to be summable with the conv branch's output. "
    "A 1×1 conv operates per-pixel and preserves `(H, W)` (modulo "
    "stride). `nn.Linear` would require flattening and lose the spatial "
    "axis.\n"
    "\n"
    "**Why no bias on a typical 1×1 shortcut.** In real ResNet, the "
    "shortcut conv is followed by BatchNorm — the BN's `beta` absorbs "
    "any constant offset, so the conv's bias is redundant. This toy "
    "drill omits BN, so a default bias is fine."
)

RECAP_NO_STEM_CONTRAST = (
    "## Stem block vs single-conv downsample — output shape contrast\n"
    "\n"
    "Ex1 BUILT the 4-op ResNet stem (Conv 7×7 s=2 → BN → ReLU → MaxPool "
    "3×3 s=2) and verified `(B, 3, 224, 224) → (B, 64, 56, 56)`. The "
    "deepening move is to compare against a NAIVE alternative — a single "
    "Conv 3×3 stride 2 — and show the output shapes diverge.\n"
    "\n"
    "**Naive alternative:** `nn.Conv2d(3, 64, kernel_size=3, stride=2, "
    "padding=1)`. One operation. Halves spatial dims once.\n"
    "\n"
    "Shape math (input 224×224):\n"
    "- ResNet stem: 224 → 112 (Conv s=2) → 56 (MaxPool s=2). **Final: "
    "56×56.**\n"
    "- Naive 3×3 s=2: `(224 + 2 - 3) // 2 + 1 = 112`. **Final: 112×112.**\n"
    "\n"
    "**Why ResNet does TWO downsamples in the stem.** A 4× total "
    "reduction up front cuts downstream FLOPs by 16× (in the spatial "
    "axes). The cost is information loss in the first layers — but the "
    "later residual blocks recover representational capacity with depth.\n"
    "\n"
    "**Receptive field difference.** The 7×7 conv sees a much larger "
    "input patch per output pixel than 3×3. Combined with the MaxPool's "
    "non-overlapping 2×2 windows (effectively, given stride 2 padding 1), "
    "early-layer features encode larger image regions — useful for "
    "natural-image tasks but overkill for tiny inputs like CIFAR.\n"
    "\n"
    "**Channel growth is the same.** Both designs go `3 → 64` — channels "
    "are decoupled from the spatial reduction strategy."
)

RECAP_STRICT_TRUE_RAISES = (
    "## `strict=True` raises on missing keys — catch + report\n"
    "\n"
    "Ex1 used `strict=False` and inspected `_IncompatibleKeys`. The "
    "deepening move is to call with `strict=True` (the default) on a "
    "checkpoint that's missing some keys, catch the `RuntimeError`, and "
    "extract the missing key names from its message.\n"
    "\n"
    "```python\n"
    "try:\n"
    "    model.load_state_dict(checkpoint)  # strict=True default\n"
    "except RuntimeError as e:\n"
    "    msg = str(e)\n"
    "    # Parse 'Missing key(s) in state_dict:' section.\n"
    "```\n"
    "\n"
    "**Why catch instead of preventing.** If you control the checkpoint, "
    "you'd never see this. But when loading a CHECKPOINT THAT EVOLVED "
    "with the model (added new layers, renamed parameters), a "
    "`strict=True` load is the canary that warns you the checkpoint is "
    "stale before you train on stale weights.\n"
    "\n"
    "**Format of the error message.** The `RuntimeError` text contains "
    "two sections: `'Missing key(s) in state_dict: \"key1\", \"key2\", "
    "...'` and (separately) `'Unexpected key(s) in state_dict: ...'`. The "
    "quoted keys are comma-separated. Parsing them is a regex job, but "
    "for a small drill you can split on the markers.\n"
    "\n"
    "**Trade-off vs `strict=False`.** `strict=False` returns the lists "
    "directly (cleaner), but silently allows mismatches if you forget to "
    "inspect the return. `strict=True` forces the caller to handle "
    "mismatches explicitly — the right default for production code."
)

RECAP_CPU_NOOP_PATH = (
    "## `torch.cuda.empty_cache()` on CPU-only — it's a safe no-op\n"
    "\n"
    "Ex1 patched `empty_cache` with a mock to count calls. The deepening "
    "move: on a CPU-only machine (no GPU), `torch.cuda.empty_cache()` "
    "IS A NO-OP. It does not raise. You can ship a single code path that "
    "calls it unconditionally.\n"
    "\n"
    "```python\n"
    "torch.cuda.empty_cache()       # CPU-only: returns None, no error\n"
    "torch.cuda.is_available()      # CPU-only: False\n"
    "```\n"
    "\n"
    "**Why this matters in practice.** Training code with GPU "
    "instrumentation should still run on the CI box and the laptop. "
    "Wrapping every CUDA call in `if cuda.is_available():` clutters the "
    "code; relying on the silent no-op (where PyTorch provides one) "
    "keeps the call sites clean.\n"
    "\n"
    "**Not ALL CUDA APIs are no-ops on CPU.** `cuda.synchronize()` also "
    "no-ops on CPU. But `torch.tensor([1]).cuda()` raises "
    "`AssertionError` because there's no device to send to. The "
    "is-available check is mandatory for OPS that require GPU memory; "
    "optional for ones that PyTorch made degenerate-safe.\n"
    "\n"
    "**Drill semantics.** This deepening drill walks the CPU code path "
    "explicitly: assert `is_available()` returns False (on the runner), "
    "then call `empty_cache()` and confirm no exception. The mock "
    "approach from ex1 is gone — we exercise the REAL function."
)

RECAP_DETACH_VS_CLONE = (
    "## `.detach()` shares storage; `.detach().clone()` is independent\n"
    "\n"
    "Ex1 built the full `.detach().clone()` snapshot. The deepening move "
    "is to CONTRAST the two operations — show that `.detach()` ALONE "
    "shares storage with the source.\n"
    "\n"
    "```python\n"
    "x = t.tensor([1.0, 2.0, 3.0], requires_grad=True)\n"
    "shared = x.detach()             # severs graph; storage is the SAME\n"
    "snap = x.detach().clone()       # severs graph AND copies storage\n"
    "x.data[0] = 99.0\n"
    "shared[0]  # → 99.0   (storage shared)\n"
    "snap[0]    # → 1.0    (independent copy)\n"
    "```\n"
    "\n"
    "**Why `.detach()` shares storage by design.** It's a graph "
    "operation, not a memory operation. The whole point is 'view of the "
    "same tensor, minus the autograd graph link'. If you wanted a new "
    "buffer, you'd ask for one — that's what `.clone()` is for.\n"
    "\n"
    "**Why this is a common bug.** `model.weight.detach()` looks like a "
    "snapshot — but if anyone later writes into `model.weight.data`, the "
    "'snapshot' updates too. The mental model 'detach() = copy' is "
    "wrong. The right model: 'detach() = read-only graph cut; clone() = "
    "make a real copy'.\n"
    "\n"
    "**Sharing flag — `.data_ptr()`.** Two tensors share storage iff "
    "they have the same `.data_ptr()` (and overlapping byte ranges). "
    "This is the load-bearing assertion for this drill."
)

RECAP_TWO_INDEX_TENSORS = (
    "## Advanced indexing with two index tensors (rows + cols)\n"
    "\n"
    "Ex1 used ONE index tensor for the leading axis (`embed[idx]`). The "
    "deepening move is two index tensors broadcasting against each "
    "other for explicit gather:\n"
    "\n"
    "```python\n"
    "x = t.arange(20).reshape(4, 5)              # (4, 5)\n"
    "rows = t.tensor([0, 2, 3])                  # (3,)\n"
    "cols = t.tensor([1, 4, 0])                  # (3,)\n"
    "x[rows, cols]                                # (3,) — diagonal pick\n"
    "# → tensor([x[0,1], x[2,4], x[3,0]])\n"
    "```\n"
    "\n"
    "**Both index tensors broadcast together to form the OUTPUT shape.** "
    "When `rows` and `cols` are both shape `(K,)`, the result is shape "
    "`(K,)` and you get K paired picks. To produce a `(R, C)` grid "
    "instead, reshape: `x[rows[:, None], cols[None, :]]`.\n"
    "\n"
    "**Why advanced indexing over `gather`.** `gather(dim, idx)` "
    "requires the index tensor to broadcast against the input on ALL "
    "axes except `dim`. Multi-axis advanced indexing has cleaner "
    "semantics when you have one index per output element — it's the "
    "'pick K specific cells' operation in shorthand.\n"
    "\n"
    "**Shape rule recap.** Multiple index tensors → output shape is the "
    "broadcast of those index tensors' shapes. The original axes being "
    "indexed COLLAPSE. So `(4, 5)` indexed by two `(K,)` tensors gives "
    "a `(K,)` output, not `(K, 5)`."
)

RECAP_BATCHED_MATVEC = (
    "## Batched matvec — `bmm` vs `einsum('bij,bj->bi', A, x)`\n"
    "\n"
    "Ex1 did single-sample matvec `W @ x`. The deepening move is the "
    "BATCHED variant: `(B, M, N) @ (B, N) → (B, M)`. Two equivalent "
    "expressions:\n"
    "\n"
    "```python\n"
    "# Option 1: torch.bmm — but bmm wants (B, M, N) @ (B, N, 1) → (B, M, 1)\n"
    "y_bmm = t.bmm(A, x.unsqueeze(-1)).squeeze(-1)   # (B, M)\n"
    "\n"
    "# Option 2: einsum — direct shape declaration, no reshape\n"
    "y_einsum = t.einsum('bij,bj->bi', A, x)         # (B, M)\n"
    "```\n"
    "\n"
    "**`bmm` requires 3-D × 3-D.** You can't do `bmm(A, x)` when `x` is "
    "2-D `(B, N)` — bmm strictly requires `(B, N, K)` on the right and "
    "produces `(B, M, K)`. The matvec emerges as the `K=1` special case "
    "with explicit `unsqueeze` + `squeeze`.\n"
    "\n"
    "**einsum just declares the shape.** `'bij,bj->bi'` says: batch "
    "shared on `b`; contract over `j`; output is `(b, i)`. No reshape "
    "scaffolding — the contraction pattern IS the function signature.\n"
    "\n"
    "**Why both are useful to know.** `bmm` is a single fast kernel — "
    "good for inner loops. `einsum` is readable for diverse contraction "
    "patterns. Production code mixes them: bmm for hot paths, einsum "
    "for one-off transformations.\n"
    "\n"
    "**Numerical equivalence.** Both compute the same FLOPs in the same "
    "order — `allclose` with default tolerance succeeds. Differences "
    "only appear with mixed dtypes (`bmm` may downcast intermediate "
    "accumulators on some backends; einsum has its own rules)."
)


# ---------------------------------------------------------------------------
# SPEC 1 — replace-final-head ex2
# ---------------------------------------------------------------------------

SPEC_FREEZE_HEAD = {
    "atom_id": "replace-final-head",
    "subtopic": "Transfer: Replace final head",
    "topic_folder": TOPIC_RESNET,
    "atom_recap_md": RECAP_FREEZE_HEAD,
    "exercise_index": 2,
    "exercise_title": "freeze the backbone after swapping the head and verify only the head has gradients",
    "slug": "freeze-backbone-after-head-swap-and-verify-grad",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["transfer-learning", "freeze", "requires_grad", "backward"],
    "kcs": [
        "requires-grad-false-in-place",
        "verify-grad-only-on-new-head",
    ],
    "lo": (
        "Analyze a head-swapped model post-`backward` to confirm that "
        "after freezing the backbone with `requires_grad_(False)`, only "
        "the new head's parameters carry gradients (`.grad is not None`) "
        "while the backbone params have `.grad is None`."
    ),
    "prompt_body": (
        "Implement `ex2_swap_freeze_and_check(model, new_num_classes, x, y)`.\n\n"
        "Steps:\n"
        "1. Read `in_features = model.fc.in_features`.\n"
        "2. Replace `model.fc = nn.Linear(in_features, new_num_classes)` — "
        "the new head is automatically trainable.\n"
        "3. Freeze the backbone: for every parameter of "
        "`model.backbone`, call `p.requires_grad_(False)`.\n"
        "4. Forward `out = model(x)` then compute "
        "`loss = nn.functional.cross_entropy(out, y)`.\n"
        "5. Call `loss.backward()`.\n"
        "6. Return a dict:\n"
        "   ```\n"
        "   {\n"
        "     'head_param_names_with_grad': [names with .grad is not None],\n"
        "     'backbone_param_names_with_grad': [names with .grad is not None],\n"
        "     'head_grad_norms': {name: float(p.grad.norm())} for those params,\n"
        "   }\n"
        "   ```\n"
        "   Use `model.named_parameters()` and split on the qname prefix "
        "(`'fc.'` for head, `'backbone.'` for backbone).\n\n"
        "Expected outcome (drives the test):\n"
        "- `head_param_names_with_grad` is non-empty (head trained).\n"
        "- `backbone_param_names_with_grad` is empty (backbone frozen).\n"
        "- Every head grad norm is strictly positive."
    ),
    "stub": (
        "def ex2_swap_freeze_and_check(model, new_num_classes: int, x, y) -> dict:\n"
        '    """Swap head, freeze backbone, run backward, report which params got grads."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Toy model identical to the ex1 setup ===\n"
        "class ToyPretrained(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.backbone = nn.Sequential(\n"
        "            nn.Linear(10, 32),\n"
        "            nn.ReLU(),\n"
        "            nn.Linear(32, 16),\n"
        "        )\n"
        "        self.fc = nn.Linear(16, 1000)\n"
        "    def forward(self, x):\n"
        "        return self.fc(self.backbone(x))\n"
        "\n"
        "model = ToyPretrained()\n"
        "x = t.randn(8, 10)\n"
        "y = t.randint(0, 5, (8,))\n"
        "\n"
        "rep = ex2_swap_freeze_and_check(model, new_num_classes=5, x=x, y=y)\n"
        "\n"
        "# === Head was actually swapped ===\n"
        "assert isinstance(model.fc, nn.Linear), 'fc must remain a Linear'\n"
        "assert model.fc.out_features == 5, f'out_features should be 5, got {model.fc.out_features}'\n"
        "assert model.fc.in_features == 16, f'in_features must stay 16 (backbone width), got {model.fc.in_features}'\n"
        "\n"
        "# === Report shape: required keys present ===\n"
        "assert set(rep.keys()) >= {'head_param_names_with_grad', 'backbone_param_names_with_grad', 'head_grad_norms'}, f'missing keys: {list(rep.keys())}'\n"
        "\n"
        "# === Backbone is frozen — no params with grad ===\n"
        "assert rep['backbone_param_names_with_grad'] == [], (\n"
        "    f'backbone should have no .grad, got {rep[\"backbone_param_names_with_grad\"]}'\n"
        ")\n"
        "\n"
        "# === Head has both weight and bias with grad ===\n"
        "head_names = set(rep['head_param_names_with_grad'])\n"
        "assert head_names == {'fc.weight', 'fc.bias'}, f'expected fc.weight + fc.bias, got {head_names}'\n"
        "\n"
        "# === Each backbone param's requires_grad is now False ===\n"
        "for name, p in model.backbone.named_parameters():\n"
        "    assert p.requires_grad is False, f'backbone param {name} not frozen'\n"
        "    assert p.grad is None, f'backbone param {name} has unexpected grad: {p.grad}'\n"
        "\n"
        "# === Each head param's requires_grad is True ===\n"
        "for name, p in model.fc.named_parameters():\n"
        "    assert p.requires_grad is True, f'head param {name} should be trainable'\n"
        "    assert p.grad is not None, f'head param {name} missing grad'\n"
        "\n"
        "# === Head grad norms strictly positive ===\n"
        "for name, norm in rep['head_grad_norms'].items():\n"
        "    assert isinstance(norm, float), f'{name}: norm must be float, got {type(norm).__name__}'\n"
        "    assert norm > 0.0, f'{name}: grad norm should be > 0, got {norm}'\n"
        "\n"
        "# === Optimizer-style filter works: trainable params == head params ===\n"
        "trainable = [name for name, p in model.named_parameters() if p.requires_grad]\n"
        "assert set(trainable) == {'fc.weight', 'fc.bias'}, f'expected only head trainable, got {trainable}'\n"
        "\n"
        "# === Repeatable: a second call (fresh model) is consistent ===\n"
        "model2 = ToyPretrained()\n"
        "rep2 = ex2_swap_freeze_and_check(model2, new_num_classes=3, x=x, y=t.randint(0, 3, (8,)))\n"
        "assert rep2['backbone_param_names_with_grad'] == []\n"
        "assert set(rep2['head_param_names_with_grad']) == {'fc.weight', 'fc.bias'}\n"
        "assert model2.fc.out_features == 3"
    ),
    "solution_body": (
        "def ex2_swap_freeze_and_check(model, new_num_classes, x, y):\n"
        "    in_features = model.fc.in_features\n"
        "    model.fc = nn.Linear(in_features, new_num_classes)\n"
        "    for p in model.backbone.parameters():\n"
        "        p.requires_grad_(False)\n"
        "    out = model(x)\n"
        "    loss = nn.functional.cross_entropy(out, y)\n"
        "    loss.backward()\n"
        "    head_with_grad = []\n"
        "    backbone_with_grad = []\n"
        "    head_norms = {}\n"
        "    for name, p in model.named_parameters():\n"
        "        if name.startswith('fc.'):\n"
        "            if p.grad is not None:\n"
        "                head_with_grad.append(name)\n"
        "                head_norms[name] = float(p.grad.norm())\n"
        "        elif name.startswith('backbone.'):\n"
        "            if p.grad is not None:\n"
        "                backbone_with_grad.append(name)\n"
        "    return {\n"
        "        'head_param_names_with_grad': sorted(head_with_grad),\n"
        "        'backbone_param_names_with_grad': sorted(backbone_with_grad),\n"
        "        'head_grad_norms': head_norms,\n"
        "    }"
    ),
    "solution_notes": (
        "**`p.grad is None` is the freeze tell, not `p.grad == 0`.** When "
        "a parameter has `requires_grad=False`, autograd never builds a "
        "graph node for it, so `.grad` stays at its initial `None`. A "
        "frozen param with grad zero would mean the graph WAS built but "
        "the gradient happened to vanish — different bug.\n\n"
        "**`requires_grad_(False)` recurses through `parameters()`.** "
        "`model.backbone.parameters()` walks all leaves under backbone, "
        "regardless of nesting depth. The trailing-underscore form is "
        "the in-place idiom; you could also do `p.requires_grad = "
        "False` — same effect.\n\n"
        "**Fresh `nn.Linear` is trainable by default.** No need to set "
        "`requires_grad=True` after the swap — `nn.Parameter` defaults "
        "to `True`. Only the OLD layers needed flipping."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 2 — residual-skip-add ex2
# ---------------------------------------------------------------------------

SPEC_SKIP_CONTRAST = {
    "atom_id": "residual-skip-add",
    "subtopic": "CNN: Residual skip-connection add",
    "topic_folder": TOPIC_RESNET,
    "atom_recap_md": RECAP_SKIP_SHAPE_CONTRAST,
    "exercise_index": 2,
    "exercise_title": "compare identity vs 1×1-conv shortcut on matched and mismatched cases",
    "slug": "identity-vs-1x1-shortcut-shape-contrast",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["residual", "shortcut", "1x1-conv", "identity", "shape-contrast"],
    "kcs": [
        "identity-shortcut-zero-params",
        "1x1-shortcut-channel-projection",
    ],
    "lo": (
        "Analyze the two ResidualBlock shortcut variants by reporting "
        "the skip-branch type, its parameter count, and the shape of "
        "`block.skip(x)` for both a matched (Identity) and a mismatched "
        "(1×1 Conv) configuration."
    ),
    "prompt_body": (
        "Implement `ex2_analyze_shortcut(in_channels, out_channels, "
        "first_stride, H, W)`.\n\n"
        "Build a ResidualBlock with the EXACT same conditional-shortcut "
        "logic as ex1:\n"
        "- if `in_channels == out_channels and first_stride == 1` → "
        "`self.skip = nn.Identity()`\n"
        "- else → `self.skip = nn.Conv2d(in_channels, out_channels, "
        "kernel_size=1, stride=first_stride, padding=0)`\n"
        "\n"
        "Then run a `(2, in_channels, H, W)` random input through the "
        "skip branch and return:\n"
        "```\n"
        "{\n"
        "  'skip_type': 'identity' | 'conv1x1',\n"
        "  'skip_n_params': int,                # sum of numel over self.skip.parameters()\n"
        "  'skip_out_shape': tuple,             # tuple(block.skip(x).shape)\n"
        "  'conv_out_shape': tuple,             # tuple(block.conv(x).shape)\n"
        "  'sum_shape': tuple,                  # tuple((conv(x) + skip(x)).shape)\n"
        "}\n"
        "```\n\n"
        "Constraints:\n"
        "1. The conv branch must be `nn.Conv2d(in_channels, out_channels, "
        "kernel_size=3, stride=first_stride, padding=1)` (matches ex1).\n"
        "2. `skip_type` is `'identity'` exactly when `self.skip` is "
        "`nn.Identity`; otherwise `'conv1x1'`.\n"
        "3. Use `t.manual_seed(0)` BEFORE constructing the block AND "
        "before generating the input so the test is deterministic.\n"
        "4. Tuples (not torch.Size) — wrap with `tuple(...)`."
    ),
    "stub": (
        "def ex2_analyze_shortcut(in_channels: int, out_channels: int,\n"
        "                          first_stride: int, H: int, W: int) -> dict:\n"
        '    """Build a ResidualBlock and report skip-branch shape + param counts."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Matched case (in=out, stride=1) → Identity, 0 params, same shape ===\n"
        "rep = ex2_analyze_shortcut(in_channels=8, out_channels=8, first_stride=1, H=16, W=16)\n"
        "assert rep['skip_type'] == 'identity', f'in=out=8 stride=1 should be identity, got {rep[\"skip_type\"]}'\n"
        "assert rep['skip_n_params'] == 0, f'identity must have 0 params, got {rep[\"skip_n_params\"]}'\n"
        "assert rep['skip_out_shape'] == (2, 8, 16, 16), f'identity preserves shape, got {rep[\"skip_out_shape\"]}'\n"
        "assert rep['conv_out_shape'] == (2, 8, 16, 16)\n"
        "assert rep['sum_shape'] == (2, 8, 16, 16)\n"
        "\n"
        "# === Channel-mismatch case (in=4 out=8, stride=1) → 1x1 conv ===\n"
        "rep = ex2_analyze_shortcut(in_channels=4, out_channels=8, first_stride=1, H=16, W=16)\n"
        "assert rep['skip_type'] == 'conv1x1', f'in!=out should be conv1x1, got {rep[\"skip_type\"]}'\n"
        "# 1x1 conv from 4 -> 8 channels: weight (8, 4, 1, 1) = 32, bias (8,) = 8, total 40.\n"
        "assert rep['skip_n_params'] == 40, f'1x1 conv 4->8 has 8*4 + 8 = 40 params, got {rep[\"skip_n_params\"]}'\n"
        "assert rep['skip_out_shape'] == (2, 8, 16, 16)\n"
        "assert rep['conv_out_shape'] == (2, 8, 16, 16)\n"
        "assert rep['sum_shape'] == (2, 8, 16, 16)\n"
        "\n"
        "# === Stride-mismatch case (in=8 out=8, stride=2) → 1x1 conv, halved spatial ===\n"
        "rep = ex2_analyze_shortcut(in_channels=8, out_channels=8, first_stride=2, H=16, W=16)\n"
        "assert rep['skip_type'] == 'conv1x1', f'stride=2 should force conv1x1, got {rep[\"skip_type\"]}'\n"
        "# 1x1 conv 8->8: 8*8 + 8 = 72.\n"
        "assert rep['skip_n_params'] == 72, f'1x1 conv 8->8 has 72 params, got {rep[\"skip_n_params\"]}'\n"
        "# 16 -> 8 via stride 2 (kernel=1, padding=0): (16 + 0 - 1)//2 + 1 = 8.\n"
        "assert rep['skip_out_shape'] == (2, 8, 8, 8), f'stride 2 halves spatial; got {rep[\"skip_out_shape\"]}'\n"
        "# conv branch (3x3, stride=2, padding=1): (16 + 2 - 3)//2 + 1 = 8.\n"
        "assert rep['conv_out_shape'] == (2, 8, 8, 8)\n"
        "assert rep['sum_shape'] == (2, 8, 8, 8)\n"
        "\n"
        "# === Combined mismatch (in=4 out=16, stride=2) ===\n"
        "rep = ex2_analyze_shortcut(in_channels=4, out_channels=16, first_stride=2, H=8, W=8)\n"
        "assert rep['skip_type'] == 'conv1x1'\n"
        "# 4 -> 16 channels via 1x1: 16*4 + 16 = 80.\n"
        "assert rep['skip_n_params'] == 80\n"
        "# 8 spatial down to (8 - 1)//2 + 1 = 4.\n"
        "assert rep['skip_out_shape'] == (2, 16, 4, 4)\n"
        "assert rep['conv_out_shape'] == (2, 16, 4, 4)\n"
        "assert rep['sum_shape'] == (2, 16, 4, 4)\n"
        "\n"
        "# === Non-square spatial input still consistent ===\n"
        "rep = ex2_analyze_shortcut(in_channels=2, out_channels=2, first_stride=1, H=7, W=11)\n"
        "assert rep['skip_type'] == 'identity'\n"
        "assert rep['skip_out_shape'] == (2, 2, 7, 11)\n"
        "assert rep['sum_shape'] == (2, 2, 7, 11)\n"
        "\n"
        "# === All shape tuples are plain Python tuples, not torch.Size ===\n"
        "for k in ('skip_out_shape', 'conv_out_shape', 'sum_shape'):\n"
        "    assert type(rep[k]) is tuple, f'{k} must be plain tuple, got {type(rep[k]).__name__}'"
    ),
    "solution_body": (
        "def ex2_analyze_shortcut(in_channels, out_channels, first_stride, H, W):\n"
        "    class ResidualBlock(nn.Module):\n"
        "        def __init__(self, ic, oc, stride):\n"
        "            super().__init__()\n"
        "            self.conv = nn.Conv2d(ic, oc, kernel_size=3, stride=stride, padding=1)\n"
        "            if ic == oc and stride == 1:\n"
        "                self.skip = nn.Identity()\n"
        "            else:\n"
        "                self.skip = nn.Conv2d(ic, oc, kernel_size=1, stride=stride, padding=0)\n"
        "        def forward(self, x):\n"
        "            return self.conv(x) + self.skip(x)\n"
        "\n"
        "    t.manual_seed(0)\n"
        "    block = ResidualBlock(in_channels, out_channels, first_stride)\n"
        "    t.manual_seed(0)\n"
        "    x = t.randn(2, in_channels, H, W)\n"
        "    skip_out = block.skip(x)\n"
        "    conv_out = block.conv(x)\n"
        "    sum_out = conv_out + skip_out\n"
        "    return {\n"
        "        'skip_type': 'identity' if isinstance(block.skip, nn.Identity) else 'conv1x1',\n"
        "        'skip_n_params': sum(p.numel() for p in block.skip.parameters()),\n"
        "        'skip_out_shape': tuple(skip_out.shape),\n"
        "        'conv_out_shape': tuple(conv_out.shape),\n"
        "        'sum_shape': tuple(sum_out.shape),\n"
        "    }"
    ),
    "solution_notes": (
        "**`nn.Identity()` has zero parameters.** Iterating "
        "`Identity().parameters()` yields nothing — so `sum(p.numel() "
        "for p in ...)` is 0. Useful as a free-of-charge fallback when "
        "you want a uniform `forward(x): conv(x) + skip(x)` interface "
        "but only sometimes need real work in the skip branch.\n\n"
        "**1×1 conv params: `out_channels * in_channels + out_channels`.** "
        "Weight is `(out, in, 1, 1)` flattening to `out * in`; bias is "
        "`(out,)`. For 4→8 that's `8*4 + 8 = 40`. The bias is the "
        "default; in real ResNet you'd disable it because BN absorbs the "
        "shift.\n\n"
        "**Why seed twice.** The block-construction seed picks the "
        "conv's initial weights; the input-construction seed picks the "
        "random input. Setting `manual_seed` before each one is more "
        "predictable than relying on one seed at the top and trusting "
        "the order of consumption inside `t.randn` calls."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 3 — resnet-stem ex2
# ---------------------------------------------------------------------------

SPEC_NO_STEM_CONTRAST = {
    "atom_id": "resnet-stem",
    "subtopic": "CNN: ResNet stem block",
    "topic_folder": TOPIC_RESNET,
    "atom_recap_md": RECAP_NO_STEM_CONTRAST,
    "exercise_index": 2,
    "exercise_title": "contrast ResNet stem with a single Conv 3×3 stride 2 alternative",
    "slug": "stem-vs-single-conv-shape-contrast",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["resnet", "stem", "downsample", "shape-math", "contrast"],
    "kcs": [
        "stem-vs-naive-downsample-shape",
        "stem-param-count-vs-naive",
    ],
    "lo": (
        "Analyze the shape and parameter-count contrast between the "
        "ResNet stem (Conv 7×7 s=2 + BN + ReLU + MaxPool 3×3 s=2) and a "
        "naive single Conv 3×3 stride 2 — same input `(B, 3, 224, 224)` "
        "yields different output shapes."
    ),
    "prompt_body": (
        "Implement `ex2_compare_stem_vs_naive()`.\n\n"
        "Build TWO modules:\n"
        "\n"
        "1. `stem`: the same `nn.Sequential` as ex1:\n"
        "   - `nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)`\n"
        "   - `nn.BatchNorm2d(64)`\n"
        "   - `nn.ReLU(inplace=True)`\n"
        "   - `nn.MaxPool2d(kernel_size=3, stride=2, padding=1)`\n"
        "\n"
        "2. `naive`: a single conv:\n"
        "   - `nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)` "
        "(default bias=True).\n"
        "\n"
        "Run a `(1, 3, 224, 224)` zero input through each (use "
        "`stem.eval()` and `naive.eval()` first so BatchNorm is "
        "deterministic), then return:\n"
        "```\n"
        "{\n"
        "  'stem_out_shape': tuple,         # (1, 64, 56, 56)\n"
        "  'naive_out_shape': tuple,        # (1, 64, 112, 112)\n"
        "  'stem_n_params': int,            # sum over stem.parameters()\n"
        "  'naive_n_params': int,           # sum over naive.parameters()\n"
        "  'spatial_reduction_factor_stem': int,   # 224 // 56 = 4\n"
        "  'spatial_reduction_factor_naive': int,  # 224 // 112 = 2\n"
        "}\n"
        "```\n"
        "\n"
        "Constraints:\n"
        "- `tuple(...)` for shapes, not `torch.Size`.\n"
        "- The reduction factor is `224 // out_spatial` (integer)."
    ),
    "stub": (
        "def ex2_compare_stem_vs_naive() -> dict:\n"
        '    """Compare ResNet stem (4x downsample) vs naive 3x3 s=2 (2x downsample)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "rep = ex2_compare_stem_vs_naive()\n"
        "\n"
        "# === Required keys present ===\n"
        "needed = {'stem_out_shape', 'naive_out_shape', 'stem_n_params', 'naive_n_params',\n"
        "          'spatial_reduction_factor_stem', 'spatial_reduction_factor_naive'}\n"
        "assert set(rep.keys()) >= needed, f'missing keys: {needed - set(rep.keys())}'\n"
        "\n"
        "# === Stem output shape: (1, 64, 56, 56) ===\n"
        "assert rep['stem_out_shape'] == (1, 64, 56, 56), f'stem shape wrong: {rep[\"stem_out_shape\"]}'\n"
        "\n"
        "# === Naive output shape: (1, 64, 112, 112) ===\n"
        "assert rep['naive_out_shape'] == (1, 64, 112, 112), f'naive shape wrong: {rep[\"naive_out_shape\"]}'\n"
        "\n"
        "# === Reduction factors: stem 4x, naive 2x ===\n"
        "assert rep['spatial_reduction_factor_stem'] == 4, f'stem 4x reduction, got {rep[\"spatial_reduction_factor_stem\"]}x'\n"
        "assert rep['spatial_reduction_factor_naive'] == 2, f'naive 2x reduction, got {rep[\"spatial_reduction_factor_naive\"]}x'\n"
        "\n"
        "# === Param counts: stem has more (7x7 conv + BN), naive has the 3x3 + bias ===\n"
        "# Stem: Conv 7x7 bias=False = 3*64*49 = 9408. BN: 2 * 64 = 128. ReLU/MaxPool: 0. Total 9536.\n"
        "assert rep['stem_n_params'] == 9536, f'stem param count: expected 9536, got {rep[\"stem_n_params\"]}'\n"
        "# Naive: Conv 3x3 + bias = 3*64*9 + 64 = 1728 + 64 = 1792.\n"
        "assert rep['naive_n_params'] == 1792, f'naive param count: expected 1792, got {rep[\"naive_n_params\"]}'\n"
        "\n"
        "# === Stem uses MORE params for MORE aggressive downsample (the trade) ===\n"
        "assert rep['stem_n_params'] > rep['naive_n_params'], 'stem should be heavier than naive'\n"
        "\n"
        "# === Output spatial dims are integer powers of 2 ===\n"
        "_, _, hs, ws = rep['stem_out_shape']\n"
        "assert hs == ws == 56\n"
        "_, _, hn, wn = rep['naive_out_shape']\n"
        "assert hn == wn == 112\n"
        "\n"
        "# === Stem output has 4x fewer spatial elements than naive ===\n"
        "stem_spatial = hs * ws            # 56 * 56 = 3136\n"
        "naive_spatial = hn * wn           # 112 * 112 = 12544\n"
        "assert naive_spatial // stem_spatial == 4, f'naive has 4x more spatial elements than stem, got {naive_spatial // stem_spatial}x'\n"
        "\n"
        "# === Tuples not torch.Size ===\n"
        "for k in ('stem_out_shape', 'naive_out_shape'):\n"
        "    assert type(rep[k]) is tuple, f'{k} must be plain tuple, got {type(rep[k]).__name__}'"
    ),
    "solution_body": (
        "def ex2_compare_stem_vs_naive():\n"
        "    stem = nn.Sequential(\n"
        "        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),\n"
        "        nn.BatchNorm2d(64),\n"
        "        nn.ReLU(inplace=True),\n"
        "        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),\n"
        "    )\n"
        "    naive = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)\n"
        "    stem.eval(); naive.eval()\n"
        "    x = t.zeros(1, 3, 224, 224)\n"
        "    with t.no_grad():\n"
        "        y_stem = stem(x)\n"
        "        y_naive = naive(x)\n"
        "    return {\n"
        "        'stem_out_shape': tuple(y_stem.shape),\n"
        "        'naive_out_shape': tuple(y_naive.shape),\n"
        "        'stem_n_params': sum(p.numel() for p in stem.parameters()),\n"
        "        'naive_n_params': sum(p.numel() for p in naive.parameters()),\n"
        "        'spatial_reduction_factor_stem': 224 // int(y_stem.shape[-1]),\n"
        "        'spatial_reduction_factor_naive': 224 // int(y_naive.shape[-1]),\n"
        "    }"
    ),
    "solution_notes": (
        "**Stem param count math.** Conv 7×7 with `bias=False`: `3 * 64 "
        "* 7 * 7 = 9408`. BatchNorm2d(64) has weight + bias = `2 * 64 = "
        "128`. ReLU and MaxPool are parameter-free. Total `9408 + 128 = "
        "9536`.\n\n"
        "**Naive param count math.** Conv 3×3 with default `bias=True`: "
        "`3 * 64 * 9 + 64 = 1728 + 64 = 1792`. Smaller by ~5×.\n\n"
        "**Spatial-element ratio is the FLOPs proxy.** Stem outputs "
        "`56² = 3136` per channel; naive outputs `112² = 12544`. A "
        "factor-of-4 downstream FLOPs saving — that's why ResNet pays "
        "the stem's extra params upfront."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 4 — state-dict-load ex2
# ---------------------------------------------------------------------------

SPEC_STRICT_TRUE_RAISES = {
    "atom_id": "state-dict-load",
    "subtopic": "Transfer: state_dict load",
    "topic_folder": TOPIC_RESNET,
    "atom_recap_md": RECAP_STRICT_TRUE_RAISES,
    "exercise_index": 2,
    "exercise_title": "load_state_dict(strict=True) raises on missing keys — catch and parse",
    "slug": "load-state-dict-strict-true-catch-and-parse",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["load_state_dict", "strict-true", "RuntimeError", "missing-keys"],
    "kcs": [
        "strict-true-raises-runtime-error",
        "parse-missing-keys-from-error-message",
    ],
    "lo": (
        "Analyze the `RuntimeError` raised by `load_state_dict("
        "strict=True)` on a checkpoint with missing entries, extracting "
        "the missing key names from the exception message into a "
        "sorted list."
    ),
    "prompt_body": (
        "Implement `ex2_load_strict_and_report(model, checkpoint)`.\n\n"
        "Call `model.load_state_dict(checkpoint)` (no `strict=` kwarg — "
        "the default IS `strict=True`). The checkpoint is missing some "
        "keys, so a `RuntimeError` will be raised.\n"
        "\n"
        "Catch it and parse the missing key names. The error message "
        "contains a line of the form:\n"
        "```\n"
        "Missing key(s) in state_dict: \"fc.weight\", \"fc.bias\".\n"
        "```\n"
        "Extract the quoted keys (anything between `\"` characters on "
        "that line). Return:\n"
        "```\n"
        "{\n"
        "  'raised': True | False,\n"
        "  'missing_keys': sorted list[str],\n"
        "  'error_message': str (str(exc); empty string if no raise),\n"
        "}\n"
        "```\n"
        "\n"
        "Use a regex like `re.findall(r'\"([^\"]+)\"', message_after_marker)` "
        "to pull the keys. To avoid catching unexpected-keys quotes "
        "(which appear separately), grab the substring after `'Missing "
        "key(s) in state_dict:'` and ending at the next newline or the "
        "next section marker like `'Unexpected key(s)'`.\n"
        "\n"
        "If `load_state_dict` does NOT raise (checkpoint is complete), "
        "set `raised=False`, `missing_keys=[]`, `error_message=''`."
    ),
    "stub": (
        "import re\n"
        "\n"
        "def ex2_load_strict_and_report(model, checkpoint: dict) -> dict:\n"
        '    """Load a checkpoint strictly, catch the RuntimeError, parse missing keys."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Toy model from the ex1 setup ===\n"
        "class ToyModel(nn.Module):\n"
        "    def __init__(self, n_classes):\n"
        "        super().__init__()\n"
        "        self.backbone = nn.Linear(10, 16)\n"
        "        self.fc = nn.Linear(16, n_classes)\n"
        "    def forward(self, x):\n"
        "        return self.fc(self.backbone(x))\n"
        "\n"
        "# === Build a model + a checkpoint that's MISSING fc.weight and fc.bias ===\n"
        "model = ToyModel(5)\n"
        "good_sd = model.state_dict()\n"
        "partial_sd = {k: v for k, v in good_sd.items() if not k.startswith('fc.')}\n"
        "rep = ex2_load_strict_and_report(model, partial_sd)\n"
        "assert rep['raised'] is True, 'strict=True must raise on missing keys'\n"
        "assert sorted(rep['missing_keys']) == ['fc.bias', 'fc.weight'], f'expected [fc.bias, fc.weight], got {rep[\"missing_keys\"]}'\n"
        "assert 'Missing key' in rep['error_message'] or 'missing' in rep['error_message'].lower()\n"
        "\n"
        "# === Complete checkpoint → no raise ===\n"
        "model = ToyModel(5)\n"
        "rep = ex2_load_strict_and_report(model, good_sd)\n"
        "assert rep['raised'] is False, 'complete checkpoint should not raise'\n"
        "assert rep['missing_keys'] == [], f'no missing keys expected, got {rep[\"missing_keys\"]}'\n"
        "assert rep['error_message'] == '', f'no message when not raised, got {rep[\"error_message\"]!r}'\n"
        "\n"
        "# === Single-key missing ===\n"
        "model = ToyModel(5)\n"
        "missing_one_sd = {k: v for k, v in good_sd.items() if k != 'backbone.weight'}\n"
        "rep = ex2_load_strict_and_report(model, missing_one_sd)\n"
        "assert rep['raised'] is True\n"
        "assert rep['missing_keys'] == ['backbone.weight'], f'expected [backbone.weight], got {rep[\"missing_keys\"]}'\n"
        "\n"
        "# === Unexpected key in checkpoint also raises, but missing_keys=[] ===\n"
        "model = ToyModel(5)\n"
        "extra_sd = dict(good_sd)\n"
        "extra_sd['ghost.weight'] = t.zeros(4, 4)\n"
        "rep = ex2_load_strict_and_report(model, extra_sd)\n"
        "assert rep['raised'] is True\n"
        "# Our parser must NOT confuse unexpected keys with missing keys.\n"
        "assert 'ghost.weight' not in rep['missing_keys'], (\n"
        "    f'ghost.weight is UNEXPECTED, not MISSING; got missing={rep[\"missing_keys\"]}'\n"
        ")\n"
        "assert rep['missing_keys'] == [], f'no missing keys when only unexpected, got {rep[\"missing_keys\"]}'\n"
        "\n"
        "# === Both missing AND unexpected → only the missing list returned ===\n"
        "model = ToyModel(5)\n"
        "mixed_sd = {k: v for k, v in good_sd.items() if not k.startswith('fc.')}\n"
        "mixed_sd['ghost.bias'] = t.zeros(3)\n"
        "rep = ex2_load_strict_and_report(model, mixed_sd)\n"
        "assert rep['raised'] is True\n"
        "assert sorted(rep['missing_keys']) == ['fc.bias', 'fc.weight'], f'mixed case wrong: {rep[\"missing_keys\"]}'\n"
        "assert 'ghost.bias' not in rep['missing_keys']\n"
        "\n"
        "# === Result list is sorted ===\n"
        "assert rep['missing_keys'] == sorted(rep['missing_keys'])"
    ),
    "solution_body": (
        "import re\n"
        "\n"
        "def ex2_load_strict_and_report(model, checkpoint):\n"
        "    try:\n"
        "        model.load_state_dict(checkpoint)\n"
        "        return {'raised': False, 'missing_keys': [], 'error_message': ''}\n"
        "    except RuntimeError as e:\n"
        "        msg = str(e)\n"
        "        # Isolate the 'Missing key(s)...' section to avoid catching\n"
        "        # 'Unexpected key(s)...' quoted names.\n"
        "        missing = []\n"
        "        marker = 'Missing key(s) in state_dict:'\n"
        "        if marker in msg:\n"
        "            tail = msg.split(marker, 1)[1]\n"
        "            # Stop at the next section marker if present.\n"
        "            for stop in ('Unexpected key(s)', 'size mismatch', 'Error(s)'):\n"
        "                if stop in tail:\n"
        "                    tail = tail.split(stop, 1)[0]\n"
        "            missing = re.findall(r'\"([^\"]+)\"', tail)\n"
        "        return {\n"
        "            'raised': True,\n"
        "            'missing_keys': sorted(missing),\n"
        "            'error_message': msg,\n"
        "        }"
    ),
    "solution_notes": (
        "**Section isolation matters.** `RuntimeError`'s message can "
        "contain BOTH 'Missing key(s)' and 'Unexpected key(s)' "
        "sections, each with quoted names. Naively running "
        "`re.findall(r'\"([^\"]+)\"', msg)` would mix them. Splitting "
        "on the marker keeps the parser honest.\n\n"
        "**Why `strict=True` is the production default.** A silent "
        "`strict=False` load can train on a checkpoint that's stale, "
        "head-mismatched, or partially restored. The exception forces "
        "explicit handling — even if your handler is just "
        "`load_state_dict(ckpt, strict=False)` after logging the "
        "diagnostic.\n\n"
        "**Quoted-key parsing is fragile.** PyTorch's error format is "
        "stable but undocumented. A more robust solution uses "
        "`strict=False` first to get the `_IncompatibleKeys` "
        "structured return, then chooses to raise. The drill exercises "
        "the parse path explicitly because it's what you do when "
        "wrapping a library you don't control."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 5 — cuda-empty-cache ex2
# ---------------------------------------------------------------------------

SPEC_CPU_NOOP = {
    "atom_id": "cuda-empty-cache",
    "subtopic": "PyTorch: torch.cuda.empty_cache",
    "topic_folder": TOPIC_TENSOR,
    "atom_recap_md": RECAP_CPU_NOOP_PATH,
    "exercise_index": 2,
    "exercise_title": "torch.cuda.empty_cache() is a safe no-op on CPU-only — exercise the real call",
    "slug": "empty-cache-cpu-noop-safe-call",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["cuda", "cpu", "no-op", "is_available"],
    "kcs": [
        "is-available-cpu-false",
        "empty-cache-no-raise-on-cpu",
    ],
    "lo": (
        "Apply `torch.cuda.is_available()` as a runtime gate, then "
        "demonstrate that `torch.cuda.empty_cache()` does not raise on "
        "a CPU-only machine — making a single unconditional cleanup "
        "call portable across hardware."
    ),
    "prompt_body": (
        "Implement `ex2_cpu_safe_empty_cache(n_calls)`.\n\n"
        "Unlike ex1 (which patched the function with a mock), this "
        "drill calls the REAL `torch.cuda.empty_cache()` and asserts it "
        "doesn't raise.\n"
        "\n"
        "Steps:\n"
        "1. Read `is_avail = torch.cuda.is_available()`.\n"
        "2. Call `torch.cuda.empty_cache()` exactly `n_calls` times in a "
        "loop. Wrap each call in a `try/except Exception as e:` and "
        "record whether ANY call raised.\n"
        "3. Return:\n"
        "   ```\n"
        "   {\n"
        "     'cuda_is_available': bool,\n"
        "     'n_calls_attempted': int,\n"
        "     'any_raised': bool,\n"
        "     'first_error': str  ('' if none),\n"
        "   }\n"
        "   ```\n"
        "\n"
        "Constraints:\n"
        "- DO NOT mock `torch.cuda.empty_cache`. Call the real function.\n"
        "- DO NOT branch on `is_available` to skip the call — the whole "
        "point is that the unconditional call works.\n"
        "- DO NOT call any other CUDA API (no `cuda.synchronize`, no "
        "`.cuda()` on tensors) — only `empty_cache` is guaranteed to "
        "no-op on CPU.\n"
        "\n"
        "Drill semantics: on a CPU-only runner, `is_avail` is False, "
        "`any_raised` is False — i.e. the call is silent."
    ),
    "stub": (
        "def ex2_cpu_safe_empty_cache(n_calls: int) -> dict:\n"
        '    """Call torch.cuda.empty_cache() unconditionally; report CPU-only safety."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch\n"
        "\n"
        "# === Single call: must not raise on CPU-only ===\n"
        "rep = ex2_cpu_safe_empty_cache(n_calls=1)\n"
        "assert isinstance(rep['cuda_is_available'], bool), f'cuda_is_available must be bool, got {type(rep[\"cuda_is_available\"]).__name__}'\n"
        "assert rep['n_calls_attempted'] == 1, f'expected 1 call, got {rep[\"n_calls_attempted\"]}'\n"
        "assert rep['any_raised'] is False, f'empty_cache must not raise on CPU; first_error={rep[\"first_error\"]!r}'\n"
        "assert rep['first_error'] == '', f'no error string when nothing raised, got {rep[\"first_error\"]!r}'\n"
        "\n"
        "# === Many calls in a row: still no raise ===\n"
        "rep = ex2_cpu_safe_empty_cache(n_calls=50)\n"
        "assert rep['n_calls_attempted'] == 50\n"
        "assert rep['any_raised'] is False, f'50 calls failed: {rep[\"first_error\"]!r}'\n"
        "\n"
        "# === n_calls=0 → no calls, trivially no raise ===\n"
        "rep = ex2_cpu_safe_empty_cache(n_calls=0)\n"
        "assert rep['n_calls_attempted'] == 0\n"
        "assert rep['any_raised'] is False\n"
        "\n"
        "# === cuda_is_available is consistent with torch.cuda.is_available() ===\n"
        "rep = ex2_cpu_safe_empty_cache(n_calls=1)\n"
        "assert rep['cuda_is_available'] == torch.cuda.is_available(), 'reported is_available must match torch.cuda.is_available()'\n"
        "\n"
        "# === The function did NOT patch torch.cuda.empty_cache ===\n"
        "# (the real function should still be importable and unmocked)\n"
        "import inspect\n"
        "# Either it's a function/builtin (real) or a wrapper; just ensure it's not a MagicMock\n"
        "from unittest.mock import MagicMock\n"
        "assert not isinstance(torch.cuda.empty_cache, MagicMock), 'must not leave a mock installed on torch.cuda.empty_cache'\n"
        "\n"
        "# === Return shape is exactly the documented keys ===\n"
        "required = {'cuda_is_available', 'n_calls_attempted', 'any_raised', 'first_error'}\n"
        "assert set(rep.keys()) >= required, f'missing required keys: {required - set(rep.keys())}'"
    ),
    "solution_body": (
        "def ex2_cpu_safe_empty_cache(n_calls):\n"
        "    import torch\n"
        "    is_avail = torch.cuda.is_available()\n"
        "    any_raised = False\n"
        "    first_error = ''\n"
        "    for _ in range(n_calls):\n"
        "        try:\n"
        "            torch.cuda.empty_cache()\n"
        "        except Exception as e:\n"
        "            if not any_raised:\n"
        "                first_error = repr(e)\n"
        "            any_raised = True\n"
        "    return {\n"
        "        'cuda_is_available': is_avail,\n"
        "        'n_calls_attempted': n_calls,\n"
        "        'any_raised': any_raised,\n"
        "        'first_error': first_error,\n"
        "    }"
    ),
    "solution_notes": (
        "**`torch.cuda.empty_cache()` is degenerate-safe on CPU by "
        "design.** It releases unused blocks held by the caching "
        "allocator; on a host with no CUDA device, the allocator has "
        "nothing to release, so the function returns immediately. This "
        "is a contract — not a happy accident.\n\n"
        "**Why not wrap in `if cuda.is_available():`.** You could. But "
        "the noisy wrap repeats at every call site. Trust the no-op for "
        "APIs where PyTorch guarantees it. For ops that DO require GPU "
        "(`.cuda()`, `cuda.synchronize()` is also safe actually, but "
        "tensor.cuda() is not), explicit gating is mandatory.\n\n"
        "**`torch.cuda.is_available()` reports hardware, not import.** "
        "`import torch` works regardless of CUDA presence — only the "
        "operations that need a device fail. `is_available()` is the "
        "load-bearing runtime check."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 6 — detach-clone-snapshot ex2
# ---------------------------------------------------------------------------

SPEC_DETACH_VS_CLONE = {
    "atom_id": "detach-clone-snapshot",
    "subtopic": "PyTorch: detach + clone snapshot",
    "topic_folder": TOPIC_TENSOR,
    "atom_recap_md": RECAP_DETACH_VS_CLONE,
    "exercise_index": 2,
    "exercise_title": ".detach() shares storage vs .detach().clone() copies — contrast on in-place mutation",
    "slug": "detach-shares-storage-vs-clone-copies",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["detach", "clone", "storage", "data_ptr", "view"],
    "kcs": [
        "detach-shares-data-ptr",
        "clone-allocates-fresh-storage",
    ],
    "lo": (
        "Analyze the storage-sharing contrast between `x.detach()` and "
        "`x.detach().clone()` by comparing `data_ptr()` and observing "
        "which 'snapshot' tracks an in-place write to the source."
    ),
    "prompt_body": (
        "Implement `ex2_detach_vs_clone(x)`.\n\n"
        "Given a leaf tensor `x` with `requires_grad=True`:\n"
        "1. Take `shared = x.detach()` (graph-cut only).\n"
        "2. Take `snap = x.detach().clone()` (graph-cut + fresh "
        "storage).\n"
        "3. Mutate the FIRST element of `x` in place via "
        "`x.data[0] = 99.0`. (Cannot directly assign to a "
        "`requires_grad=True` leaf, but `.data` bypass works.)\n"
        "4. Return:\n"
        "   ```\n"
        "   {\n"
        "     'shared_data_ptr_equals_x': bool,         # True\n"
        "     'snap_data_ptr_equals_x': bool,           # False\n"
        "     'shared_first_after_mutation': float,     # 99.0\n"
        "     'snap_first_after_mutation': float,       # original x[0]\n"
        "     'shared_requires_grad': bool,             # False\n"
        "     'snap_requires_grad': bool,               # False\n"
        "     'x_requires_grad': bool,                  # True (unchanged)\n"
        "   }\n"
        "   ```\n"
        "\n"
        "Constraints:\n"
        "- Use `.data_ptr()` (not `is` or `==`) to compare storage "
        "identity.\n"
        "- Cast comparison values to plain Python `float`.\n"
        "- Do NOT call `.backward()` — there's no graph to walk; this "
        "is a pure storage/graph drill."
    ),
    "stub": (
        "def ex2_detach_vs_clone(x: Tensor) -> dict:\n"
        '    """Contrast shared-storage detach() with full-copy detach().clone()."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Typical case: shared shares storage, snap is independent ===\n"
        "x = t.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)\n"
        "x_first_before = float(x[0])\n"
        "rep = ex2_detach_vs_clone(x)\n"
        "\n"
        "assert rep['shared_data_ptr_equals_x'] is True, 'shared = x.detach() must share storage with x'\n"
        "assert rep['snap_data_ptr_equals_x'] is False, 'snap = x.detach().clone() must NOT share storage'\n"
        "assert rep['shared_first_after_mutation'] == 99.0, (\n"
        "    f'shared should track in-place mutation; got {rep[\"shared_first_after_mutation\"]}'\n"
        ")\n"
        "assert rep['snap_first_after_mutation'] == x_first_before, (\n"
        "    f'snap is a real copy; should be {x_first_before}, got {rep[\"snap_first_after_mutation\"]}'\n"
        ")\n"
        "\n"
        "# === Graph-cut: both shared and snap have requires_grad=False ===\n"
        "assert rep['shared_requires_grad'] is False\n"
        "assert rep['snap_requires_grad'] is False\n"
        "# x itself is unchanged in its graph status.\n"
        "assert rep['x_requires_grad'] is True\n"
        "\n"
        "# === x[0] is now 99.0 (mutation actually happened) ===\n"
        "assert float(x[0]) == 99.0, f'x[0] must have been mutated to 99, got {float(x[0])}'\n"
        "\n"
        "# === Larger tensor: same contract ===\n"
        "x2 = t.randn(64, requires_grad=True)\n"
        "x2_first_before = float(x2[0])\n"
        "rep2 = ex2_detach_vs_clone(x2)\n"
        "assert rep2['shared_data_ptr_equals_x'] is True\n"
        "assert rep2['snap_data_ptr_equals_x'] is False\n"
        "assert rep2['shared_first_after_mutation'] == 99.0\n"
        "assert abs(rep2['snap_first_after_mutation'] - x2_first_before) < 1e-6\n"
        "\n"
        "# === Cast types: floats are Python floats, not 0-D tensors ===\n"
        "for k in ('shared_first_after_mutation', 'snap_first_after_mutation'):\n"
        "    assert isinstance(rep[k], float), f'{k} must be plain float, got {type(rep[k]).__name__}'\n"
        "\n"
        "# === Bool flags are real bools ===\n"
        "for k in ('shared_data_ptr_equals_x', 'snap_data_ptr_equals_x', 'shared_requires_grad', 'snap_requires_grad', 'x_requires_grad'):\n"
        "    assert isinstance(rep[k], bool), f'{k} must be bool, got {type(rep[k]).__name__}'"
    ),
    "solution_body": (
        "def ex2_detach_vs_clone(x):\n"
        "    shared = x.detach()\n"
        "    snap = x.detach().clone()\n"
        "    x.data[0] = 99.0\n"
        "    return {\n"
        "        'shared_data_ptr_equals_x': shared.data_ptr() == x.data_ptr(),\n"
        "        'snap_data_ptr_equals_x': snap.data_ptr() == x.data_ptr(),\n"
        "        'shared_first_after_mutation': float(shared[0]),\n"
        "        'snap_first_after_mutation': float(snap[0]),\n"
        "        'shared_requires_grad': bool(shared.requires_grad),\n"
        "        'snap_requires_grad': bool(snap.requires_grad),\n"
        "        'x_requires_grad': bool(x.requires_grad),\n"
        "    }"
    ),
    "solution_notes": (
        "**`.detach()` is a graph operation only.** It returns a tensor "
        "that shares the same storage but is disconnected from "
        "autograd. Cheap (no copy), but means any in-place mutation to "
        "the source is visible through the detached view.\n\n"
        "**`.clone()` allocates a new buffer.** The result has its own "
        "`data_ptr()` and is independent. `.detach().clone()` does "
        "both: cut the graph AND copy storage. This is the safe "
        "snapshot for 'I want to remember this value'.\n\n"
        "**`x.data[0] = 99.0` bypasses autograd.** A direct `x[0] = "
        "99.0` on a leaf with `requires_grad=True` raises a "
        "RuntimeError because it would corrupt the graph. Mutating via "
        "`.data` is the documented escape hatch — but the SAME mutation "
        "is exactly why detach-without-clone is dangerous in practice."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 7 — index-by-tensor ex2
# ---------------------------------------------------------------------------

SPEC_TWO_INDEX_TENSORS = {
    "atom_id": "index-by-tensor",
    "subtopic": "PyTorch: index by tensor",
    "topic_folder": TOPIC_TENSOR,
    "atom_recap_md": RECAP_TWO_INDEX_TENSORS,
    "exercise_index": 2,
    "exercise_title": "advanced indexing with TWO index tensors (rows + cols) for paired gather",
    "slug": "advanced-indexing-two-index-tensors-paired-gather",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["advanced-indexing", "gather", "two-index-tensors", "broadcasting"],
    "kcs": [
        "paired-index-collapse-axes",
        "broadcasted-grid-via-newaxis",
    ],
    "lo": (
        "Apply two-index-tensor advanced indexing both in PAIRED form "
        "`x[rows, cols] → (K,)` and in GRID form `x[rows[:, None], "
        "cols[None, :]] → (R, C)`."
    ),
    "prompt_body": (
        "Implement `ex2_gather_with_two_indices(x, rows, cols)`.\n\n"
        "Inputs:\n"
        "- `x`: `(H, W)` 2-D tensor.\n"
        "- `rows`: `(K,)` LongTensor with values in `[0, H)`.\n"
        "- `cols`: `(K,)` LongTensor with values in `[0, W)`.\n"
        "\n"
        "Return a dict with TWO outputs of the same data, in two "
        "different output shapes:\n"
        "```\n"
        "{\n"
        "  'paired': x[rows, cols],                       # (K,)\n"
        "  'grid':   x[rows[:, None], cols[None, :]],     # (K, K)\n"
        "}\n"
        "```\n"
        "\n"
        "- `paired[i] == x[rows[i], cols[i]]` — paired (zip-style) "
        "indexing.\n"
        "- `grid[i, j] == x[rows[i], cols[j]]` — Cartesian-product "
        "indexing.\n"
        "\n"
        "Constraints:\n"
        "- DO NOT use a Python for-loop.\n"
        "- DO NOT use `torch.gather` — exercise the advanced-indexing "
        "syntax directly.\n"
        "- Both outputs must preserve `x.dtype`."
    ),
    "stub": (
        "def ex2_gather_with_two_indices(x: Tensor, rows: Tensor, cols: Tensor) -> dict:\n"
        '    """Return paired (K,) and grid (K, K) advanced-indexed selections."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Hand-traced reference ===\n"
        "x = t.arange(20).reshape(4, 5).float()\n"
        "rows = t.tensor([0, 2, 3])\n"
        "cols = t.tensor([1, 4, 0])\n"
        "rep = ex2_gather_with_two_indices(x, rows, cols)\n"
        "\n"
        "# Paired: x[0,1], x[2,4], x[3,0] = 1, 14, 15\n"
        "expected_paired = t.tensor([1.0, 14.0, 15.0])\n"
        "assert rep['paired'].shape == (3,), f'paired shape wrong: {tuple(rep[\"paired\"].shape)}'\n"
        "assert t.equal(rep['paired'], expected_paired), f'paired wrong: got {rep[\"paired\"]}, expected {expected_paired}'\n"
        "\n"
        "# Grid: 3x3 where grid[i,j] = x[rows[i], cols[j]]\n"
        "expected_grid = t.tensor([\n"
        "    [x[0,1], x[0,4], x[0,0]],\n"
        "    [x[2,1], x[2,4], x[2,0]],\n"
        "    [x[3,1], x[3,4], x[3,0]],\n"
        "])\n"
        "assert rep['grid'].shape == (3, 3), f'grid shape wrong: {tuple(rep[\"grid\"].shape)}'\n"
        "assert t.equal(rep['grid'], expected_grid), f'grid wrong: got {rep[\"grid\"]}, expected {expected_grid}'\n"
        "\n"
        "# === Single-element rows/cols ===\n"
        "rows = t.tensor([2])\n"
        "cols = t.tensor([3])\n"
        "rep = ex2_gather_with_two_indices(x, rows, cols)\n"
        "assert rep['paired'].shape == (1,)\n"
        "assert rep['paired'].item() == x[2, 3].item()\n"
        "assert rep['grid'].shape == (1, 1)\n"
        "assert rep['grid'][0, 0].item() == x[2, 3].item()\n"
        "\n"
        "# === Identity-index: rows=cols=arange recovers the diagonal ===\n"
        "x_sq = t.arange(16).reshape(4, 4).float()\n"
        "diag_idx = t.arange(4)\n"
        "rep = ex2_gather_with_two_indices(x_sq, diag_idx, diag_idx)\n"
        "assert t.equal(rep['paired'], t.diagonal(x_sq)), f'diagonal pick wrong: {rep[\"paired\"]} vs {t.diagonal(x_sq)}'\n"
        "# Grid should be the full 4x4 (all rows × all cols), which IS x_sq itself.\n"
        "assert t.equal(rep['grid'], x_sq), 'grid with diag_idx,diag_idx should reproduce x'\n"
        "\n"
        "# === Larger random check vs explicit comparison ===\n"
        "rng = t.Generator().manual_seed(42)\n"
        "big = t.randn(20, 15, generator=rng)\n"
        "rows = t.tensor([0, 5, 10, 15, 19])\n"
        "cols = t.tensor([1, 3, 7, 11, 14])\n"
        "rep = ex2_gather_with_two_indices(big, rows, cols)\n"
        "for i in range(5):\n"
        "    assert rep['paired'][i].item() == big[rows[i].item(), cols[i].item()].item(), f'paired[{i}] mismatch'\n"
        "for i in range(5):\n"
        "    for j in range(5):\n"
        "        assert rep['grid'][i, j].item() == big[rows[i].item(), cols[j].item()].item(), f'grid[{i},{j}] mismatch'\n"
        "\n"
        "# === Dtype preservation: int input → int output ===\n"
        "x_int = t.arange(20).reshape(4, 5)\n"
        "rep = ex2_gather_with_two_indices(x_int, t.tensor([0, 2]), t.tensor([1, 3]))\n"
        "assert rep['paired'].dtype == x_int.dtype, f'paired dtype must match input, got {rep[\"paired\"].dtype}'\n"
        "assert rep['grid'].dtype == x_int.dtype, f'grid dtype must match input, got {rep[\"grid\"].dtype}'\n"
        "\n"
        "# === Repeated index allowed ===\n"
        "rep = ex2_gather_with_two_indices(x, t.tensor([1, 1, 1]), t.tensor([0, 0, 0]))\n"
        "assert t.equal(rep['paired'], t.full((3,), x[1, 0].item()))"
    ),
    "solution_body": (
        "def ex2_gather_with_two_indices(x, rows, cols):\n"
        "    return {\n"
        "        'paired': x[rows, cols],\n"
        "        'grid': x[rows[:, None], cols[None, :]],\n"
        "    }"
    ),
    "solution_notes": (
        "**Paired vs grid is a SHAPE decision, not a values decision.** "
        "Same input tensors. Different bracket syntax. The first "
        "broadcasts `rows` and `cols` together (both `(K,)` → output "
        "`(K,)`). The second reshapes them into `(K, 1)` and `(1, K)` "
        "so they broadcast to `(K, K)` — and the output picks up that "
        "shape.\n\n"
        "**`rows[:, None]` is `rows.unsqueeze(1)`.** Same operation, "
        "shorter notation. Use whichever your team's style guide "
        "prefers. The `None` form reads more like NumPy.\n\n"
        "**Why this is the canonical bilinear-interp building block.** "
        "Image warping (grid_sample, STN, RoI pooling) all reduce to "
        "'gather these four corner pixels per output pixel'. The "
        "`x[rows[:, None], cols[None, :]]` pattern is the rank-2 "
        "version; rank-4 extensions use the same idea with more "
        "broadcasting axes."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 8 — matvec ex2
# ---------------------------------------------------------------------------

SPEC_BATCHED_MATVEC = {
    "atom_id": "matvec",
    "subtopic": "PyTorch: matrix-vector product",
    "topic_folder": TOPIC_TENSOR,
    "atom_recap_md": RECAP_BATCHED_MATVEC,
    "exercise_index": 2,
    "exercise_title": "batched matvec two ways: torch.bmm with unsqueeze/squeeze vs einsum 'bij,bj->bi'",
    "slug": "batched-matvec-bmm-vs-einsum",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["batched-matvec", "bmm", "einsum", "shape-discipline"],
    "kcs": [
        "bmm-rank3-by-rank3-only",
        "einsum-shape-declaration",
    ],
    "lo": (
        "Apply two equivalent batched-matvec expressions — "
        "`torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)` and "
        "`torch.einsum('bij,bj->bi', A, x)` — that both turn "
        "`(B, M, N) × (B, N) → (B, M)`, then verify they agree "
        "numerically."
    ),
    "prompt_body": (
        "Implement `ex2_batched_matvec(A, x)`.\n\n"
        "Inputs:\n"
        "- `A`: `(B, M, N)` float tensor.\n"
        "- `x`: `(B, N)` float tensor.\n"
        "\n"
        "Compute the batched matrix-vector product TWO WAYS and return "
        "both, plus a numerical equality flag:\n"
        "```\n"
        "{\n"
        "  'y_bmm': (B, M),     # torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)\n"
        "  'y_einsum': (B, M),  # torch.einsum('bij,bj->bi', A, x)\n"
        "  'allclose': bool,    # True — both methods agree (atol=1e-6)\n"
        "}\n"
        "```\n"
        "\n"
        "Constraints:\n"
        "1. `y_bmm` MUST use `torch.bmm` (not `matmul`, not `@`). The "
        "drill is about the rank-3 requirement of `bmm`.\n"
        "2. `y_einsum` MUST use `torch.einsum` with the exact spec "
        "string `'bij,bj->bi'`.\n"
        "3. DO NOT use a Python for-loop.\n"
        "4. Output dtype matches input.\n"
        "5. Both outputs are shape `(B, M)` — NOT `(B, M, 1)`."
    ),
    "stub": (
        "def ex2_batched_matvec(A: Tensor, x: Tensor) -> dict:\n"
        '    """Compute (B, M, N) @ (B, N) -> (B, M) via bmm AND einsum; verify agreement."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Small hand-traced case ===\n"
        "A = t.tensor([\n"
        "    [[1.0, 2.0], [3.0, 4.0]],   # batch 0: 2x2\n"
        "    [[0.0, 1.0], [1.0, 0.0]],   # batch 1: swap matrix\n"
        "]).float()                       # (2, 2, 2)\n"
        "x = t.tensor([\n"
        "    [1.0, 1.0],                  # batch 0: ones\n"
        "    [5.0, 7.0],                  # batch 1: (5, 7)\n"
        "])                               # (2, 2)\n"
        "rep = ex2_batched_matvec(A, x)\n"
        "# Batch 0: [1+2, 3+4] = [3, 7]\n"
        "# Batch 1: [0+7, 5+0] = [7, 5]\n"
        "expected = t.tensor([[3.0, 7.0], [7.0, 5.0]])\n"
        "assert rep['y_bmm'].shape == (2, 2), f'y_bmm shape: {tuple(rep[\"y_bmm\"].shape)}'\n"
        "assert rep['y_einsum'].shape == (2, 2), f'y_einsum shape: {tuple(rep[\"y_einsum\"].shape)}'\n"
        "assert t.allclose(rep['y_bmm'], expected), f'y_bmm wrong: {rep[\"y_bmm\"]} vs {expected}'\n"
        "assert t.allclose(rep['y_einsum'], expected), f'y_einsum wrong: {rep[\"y_einsum\"]} vs {expected}'\n"
        "assert rep['allclose'] is True\n"
        "\n"
        "# === Random batch: bmm and einsum agree ===\n"
        "rng = t.Generator().manual_seed(0)\n"
        "A = t.randn(8, 5, 4, generator=rng)   # (B=8, M=5, N=4)\n"
        "x = t.randn(8, 4, generator=rng)      # (B=8, N=4)\n"
        "rep = ex2_batched_matvec(A, x)\n"
        "assert rep['y_bmm'].shape == (8, 5)\n"
        "assert rep['y_einsum'].shape == (8, 5)\n"
        "assert t.allclose(rep['y_bmm'], rep['y_einsum'], atol=1e-6), 'bmm and einsum must agree numerically'\n"
        "assert rep['allclose'] is True\n"
        "\n"
        "# === Manual verification vs explicit per-batch matvec ===\n"
        "for b in range(8):\n"
        "    expected_b = A[b] @ x[b]\n"
        "    assert t.allclose(rep['y_bmm'][b], expected_b, atol=1e-6), f'batch {b}: bmm vs per-sample disagree'\n"
        "    assert t.allclose(rep['y_einsum'][b], expected_b, atol=1e-6), f'batch {b}: einsum vs per-sample disagree'\n"
        "\n"
        "# === Singleton batch B=1 still produces (1, M) ===\n"
        "A = t.randn(1, 3, 7)\n"
        "x = t.randn(1, 7)\n"
        "rep = ex2_batched_matvec(A, x)\n"
        "assert rep['y_bmm'].shape == (1, 3)\n"
        "assert rep['y_einsum'].shape == (1, 3)\n"
        "assert rep['allclose'] is True\n"
        "\n"
        "# === Dtype preservation ===\n"
        "A = t.randn(2, 3, 3, dtype=t.float64)\n"
        "x = t.randn(2, 3, dtype=t.float64)\n"
        "rep = ex2_batched_matvec(A, x)\n"
        "assert rep['y_bmm'].dtype == t.float64, f'bmm dtype: {rep[\"y_bmm\"].dtype}'\n"
        "assert rep['y_einsum'].dtype == t.float64, f'einsum dtype: {rep[\"y_einsum\"].dtype}'\n"
        "\n"
        "# === Outputs are 2-D not 3-D ===\n"
        "assert rep['y_bmm'].dim() == 2, f'y_bmm must be 2-D, got {rep[\"y_bmm\"].dim()}-D shape {tuple(rep[\"y_bmm\"].shape)}'\n"
        "assert rep['y_einsum'].dim() == 2, f'y_einsum must be 2-D, got {rep[\"y_einsum\"].dim()}-D shape {tuple(rep[\"y_einsum\"].shape)}'\n"
        "\n"
        "# === Larger batch ===\n"
        "A = t.randn(32, 10, 8)\n"
        "x = t.randn(32, 8)\n"
        "rep = ex2_batched_matvec(A, x)\n"
        "assert rep['y_bmm'].shape == (32, 10)\n"
        "assert rep['y_einsum'].shape == (32, 10)\n"
        "assert rep['allclose']"
    ),
    "solution_body": (
        "def ex2_batched_matvec(A, x):\n"
        "    y_bmm = t.bmm(A, x.unsqueeze(-1)).squeeze(-1)\n"
        "    y_einsum = t.einsum('bij,bj->bi', A, x)\n"
        "    return {\n"
        "        'y_bmm': y_bmm,\n"
        "        'y_einsum': y_einsum,\n"
        "        'allclose': bool(t.allclose(y_bmm, y_einsum, atol=1e-6)),\n"
        "    }"
    ),
    "solution_notes": (
        "**`bmm` is the rank-3 by rank-3 batched matmul.** It rejects "
        "rank-2 inputs. The `unsqueeze(-1)` adds a trailing dim of 1 to "
        "`x` (turning `(B, N)` into `(B, N, 1)`) so the contraction "
        "shapes line up: `(B, M, N) @ (B, N, 1) → (B, M, 1)`. The "
        "trailing `squeeze(-1)` removes that artificial axis.\n\n"
        "**einsum declares shape via the spec string.** `'bij,bj->bi'` "
        "reads: 'A has axes b, i, j; x has axes b, j; output has b, i'. "
        "The shared `b` is broadcast/batched. The shared `j` is "
        "contracted (summed over). The unique `i` becomes the output "
        "axis. No reshape scaffolding required.\n\n"
        "**Why both — pick by context.** `bmm` is a single fast kernel "
        "in CUDA — preferred for hot training loops. `einsum` is more "
        "expressive — preferred for one-off transformations where "
        "readability matters more than the last 5% perf. Both compile "
        "to the same hardware kernel under modern PyTorch."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# All specs
# ---------------------------------------------------------------------------

SPECS = [
    SPEC_FREEZE_HEAD,
    SPEC_SKIP_CONTRAST,
    SPEC_NO_STEM_CONTRAST,
    SPEC_STRICT_TRUE_RAISES,
    SPEC_CPU_NOOP,
    SPEC_DETACH_VS_CLONE,
    SPEC_TWO_INDEX_TENSORS,
    SPEC_BATCHED_MATVEC,
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
    print(f"[deepening_z_batch12] Verifying {len(SPECS)} specs...")
    _verify_all(SPECS)

    print(f"\n[deepening_z_batch12] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_z_batch12] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
