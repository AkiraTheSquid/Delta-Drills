"""Composite drills cx25..cx30 — batch-18 part2 (EE-cell, ARENA CNNs).

Six composite procedural drills exercising 2-atom pairs from ARENA part 2 —
ResNet / BatchNorm / Init machinery.

cx25  residual-skip-add + nn-module-subclass    — residual block via subclass
cx26  residual-skip-add + module-composition    — residual within sequential
cx27  conv-stride-downsample + residual-skip-add — stride-2 skip with downsample
cx28  batchnorm-affine-params + batchnorm-running-stats — full BN forward
cx29  inference-mode-step + train-eval-mode-branch — eval vs train BN paths
cx30  kaiming-uniform-sf-init + nn-parameter-wrap — init + Parameter wrap
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_composite import emit_composite  # noqa: E402

INV = {a["atom_id"]: a for a in json.load(open("/tmp/drill_atoms.json"))}


def _subs(atom_ids):
    return [INV[a]["subtopic"] for a in atom_ids]


# Standard nn imports — every spec needs these since they're all CNN-flavored.
NN_IMPORTS = [
    "import torch.nn as nn",
    "import torch.nn.functional as F",
]


# ===========================================================================
# cx25 — residual block via nn.Module subclass
# ===========================================================================
spec_25 = {
    "atom_ids": ["residual-skip-add", "nn-module-subclass"],
    "subtopics": _subs(["residual-skip-add", "nn-module-subclass"]),
    "primary_atom": "residual-skip-add",
    "part": "part2",
    "exercise_index": 25,
    "exercise_title": "residual block as an nn.Module subclass",
    "slug": "residual-block-via-module-subclass",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "A ResNet **residual block** has two ingredients:\n"
        "1. A 'main path' — usually `conv -> bn -> relu -> conv -> bn`.\n"
        "2. A skip connection that ADDS the input back to the main-path output BEFORE the final "
        "ReLU: `out = relu(main(x) + x)`.\n\n"
        "The canonical way to package this in PyTorch is to **subclass `nn.Module`**. The subclass "
        "registers its child modules in `__init__` (so `.parameters()` finds their weights, and "
        "`.to(device)` moves them) and implements the residual-add in `forward`.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "class ResidualBlock(nn.Module):\n"
        "    def __init__(self, channels):\n"
        "        super().__init__()                  # MUST call — wires up the registry.\n"
        "        self.conv1 = nn.Conv2d(...)\n"
        "        self.conv2 = nn.Conv2d(...)\n"
        "        # ... bn layers ...\n"
        "\n"
        "    def forward(self, x):\n"
        "        identity = x                         # save the skip.\n"
        "        out = self.conv2(F.relu(self.conv1(x)))\n"
        "        return F.relu(out + identity)        # residual-skip-add HERE.\n"
        "```\n\n"
        "**Why both atoms together.** The skip-add is meaningless without the module to host it. "
        "The subclass is the smallest unit that PyTorch's optimizer, `.train()/.eval()`, and "
        "`.to(device)` all hook into."
    ),
    "prompt_body": (
        "Implement a class `ResidualBlock(nn.Module)` so that for a square `(N, C, H, W)` input:\n\n"
        "- `__init__(self, channels)` calls `super().__init__()` and registers two convs:\n"
        "  - `self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)`\n"
        "  - `self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)`\n"
        "- `forward(self, x)` computes `h = F.relu(self.conv1(x))`, then "
        "`out = self.conv2(h)`, then **adds the skip `x`**, then applies a final `F.relu`.\n\n"
        "Return the constructed class (not an instance) from `cx25_make_residual_block()`.\n\n"
        "The test instantiates it, checks `super().__init__()` was called (otherwise "
        "`list(block.parameters())` would be empty), and checks the residual-add by ZEROING the "
        "conv weights — with zero convs the output must equal `relu(x)`."
    ),
    "stub_body": (
        "def cx25_make_residual_block():\n"
        "    \"\"\"Return the ResidualBlock class.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "ResidualBlock = cx25_make_residual_block()\n"
        "assert isinstance(ResidualBlock, type) and issubclass(ResidualBlock, nn.Module), (\n"
        "    'cx25 must return a class that subclasses nn.Module'\n"
        ")\n"
        "\n"
        "# Case A: instantiation & parameter registration (proves super().__init__() was called).\n"
        "t.manual_seed(0)\n"
        "block = ResidualBlock(channels=4)\n"
        "params = list(block.parameters())\n"
        "assert len(params) >= 2, (\n"
        "    f'block has only {len(params)} parameters — did you forget super().__init__()?'\n"
        ")\n"
        "# conv1 and conv2 must be registered as submodules.\n"
        "named = dict(block.named_children())\n"
        "assert 'conv1' in named and 'conv2' in named, f'expected conv1, conv2 children; got {list(named)}'\n"
        "assert isinstance(named['conv1'], nn.Conv2d) and isinstance(named['conv2'], nn.Conv2d)\n"
        "\n"
        "# Case B: shape contract — (N, C, H, W) in, (N, C, H, W) out (padding=1 + k=3 preserves H,W).\n"
        "x = t.randn(2, 4, 5, 7)\n"
        "out = block(x)\n"
        "assert tuple(out.shape) == (2, 4, 5, 7), f'expected (2,4,5,7), got {tuple(out.shape)}'\n"
        "\n"
        "# Case C: residual-skip-add is the LOAD-BEARING op.\n"
        "# Zero the conv weights — main path becomes 0, so out = relu(0 + x) = relu(x).\n"
        "with t.no_grad():\n"
        "    block.conv1.weight.zero_()\n"
        "    block.conv2.weight.zero_()\n"
        "out = block(x)\n"
        "expected = F.relu(x)  # If skip-add is missing, this would be ~0.\n"
        "assert t.allclose(out, expected, atol=1e-6), (\n"
        "    'with zeroed convs out should equal relu(x) — did you add the skip?'\n"
        ")\n"
        "# Sanity: out is NOT all-zero (would happen if the skip were missing).\n"
        "assert out.abs().sum().item() > 0, 'out is all-zero — skip connection missing'\n"
        "\n"
        "# Case D: residual-add is BEFORE the final relu (not after).\n"
        "# Build x with negative entries that, after adding to zero main-path, still get relu'd.\n"
        "x_neg = -t.ones(1, 4, 3, 3)\n"
        "out = block(x_neg)\n"
        "# relu(0 + (-1)) = 0 — every entry should be 0.\n"
        "assert t.allclose(out, t.zeros_like(out)), 'final relu should clip negatives — got nonzero'"
    ),
    "solution_body": (
        "def cx25_make_residual_block():\n"
        "    class ResidualBlock(nn.Module):\n"
        "        def __init__(self, channels):\n"
        "            # Atom B (nn-module-subclass): super().__init__() wires up the module registry.\n"
        "            super().__init__()\n"
        "            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)\n"
        "            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)\n"
        "\n"
        "        def forward(self, x):\n"
        "            identity = x\n"
        "            h = F.relu(self.conv1(x))\n"
        "            out = self.conv2(h)\n"
        "            # Atom A (residual-skip-add): add the skip BEFORE the final activation.\n"
        "            return F.relu(out + identity)\n"
        "\n"
        "    return ResidualBlock"
    ),
    "solution_notes": (
        "The `super().__init__()` call is non-negotiable — without it the `nn.Module` base never "
        "initialises its parameter/module dicts, and `block.parameters()` returns nothing. The "
        "`out + identity` is the canonical residual-add: same shape on both sides because conv "
        "with `padding=1, kernel=3` is shape-preserving. The final `F.relu` AFTER the add is the "
        "ResNet-v1 ordering — v2 moves the activation before the add (pre-activation), but v1 is "
        "the ARENA default."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["residual-skip-add", "nn-module-subclass"],
    "lo": (
        "Compose nn.Module subclassing (registry of conv children + forward override) with "
        "residual-skip-add (out + identity before final ReLU) to build the canonical ResNet "
        "residual block."
    ),
}


# ===========================================================================
# cx26 — residual block wrapping an nn.Sequential main path
# ===========================================================================
spec_26 = {
    "atom_ids": ["residual-skip-add", "module-composition"],
    "subtopics": _subs(["residual-skip-add", "module-composition"]),
    "primary_atom": "residual-skip-add",
    "part": "part2",
    "exercise_index": 26,
    "exercise_title": "residual-skip-add wrapping an nn.Sequential main path",
    "slug": "residual-skip-within-sequential",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Real ResNet blocks are often refactored so the **main path** is one `nn.Sequential` and "
        "the **skip-add** is the only thing the outer module's `forward` does explicitly. This "
        "isolates the architectural choice (the layer stack) from the residual wiring.\n\n"
        "**Module composition** is PyTorch's term for *building bigger modules out of smaller ones*. "
        "Three idioms show up:\n"
        "- `nn.Sequential(*layers)` — a list-shaped composition; forward is `layers[0](layers[1](...))`.\n"
        "- subclass with named children — flexible; you write the forward.\n"
        "- mixed — outer subclass holds an inner `nn.Sequential` for the linear stretch and adds "
        "a skip on top.\n\n"
        "The third idiom is what cx26 exercises: a residual block whose `main` attribute IS an "
        "`nn.Sequential(conv1, bn1, relu, conv2, bn2)` (no final relu inside it!), and whose "
        "`forward` is just `F.relu(self.main(x) + x)`.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "class ResBlock(nn.Module):\n"
        "    def __init__(self, c):\n"
        "        super().__init__()\n"
        "        self.main = nn.Sequential(\n"
        "            nn.Conv2d(c, c, 3, padding=1, bias=False),\n"
        "            nn.BatchNorm2d(c),\n"
        "            nn.ReLU(),\n"
        "            nn.Conv2d(c, c, 3, padding=1, bias=False),\n"
        "            nn.BatchNorm2d(c),\n"
        "        )\n"
        "    def forward(self, x):\n"
        "        return F.relu(self.main(x) + x)        # skip-add inside the wrapper.\n"
        "```\n\n"
        "**Why care.** Mixing `nn.Sequential` for the linear stretch with an outer subclass for "
        "the skip is the textbook PyTorch composition pattern. It also makes the main path a "
        "single object you can swap out (e.g. bottleneck variant) without touching the skip wiring."
    ),
    "prompt_body": (
        "Implement `cx26_make_resblock_with_sequential()` — return the class.\n\n"
        "Required structure for the returned class `ResBlock(nn.Module)`:\n"
        "- `__init__(self, channels)` calls `super().__init__()` then sets:\n"
        "  - `self.main = nn.Sequential(...)` containing, in order:\n"
        "    1. `nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)`\n"
        "    2. `nn.BatchNorm2d(channels)`\n"
        "    3. `nn.ReLU()`\n"
        "    4. `nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)`\n"
        "    5. `nn.BatchNorm2d(channels)`\n"
        "  - **No final ReLU inside `self.main`** — that goes in `forward` after the skip-add.\n"
        "- `forward(self, x)` returns `F.relu(self.main(x) + x)`.\n\n"
        "The test verifies: (a) `self.main` is an `nn.Sequential` of length 5; (b) the skip-add "
        "is load-bearing (zeroing the main-path weights yields `relu(x)`)."
    ),
    "stub_body": (
        "def cx26_make_resblock_with_sequential():\n"
        "    \"\"\"Return the ResBlock class.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "ResBlock = cx26_make_resblock_with_sequential()\n"
        "assert issubclass(ResBlock, nn.Module)\n"
        "\n"
        "# Case A: main is an nn.Sequential of exactly 5 layers, in the right order.\n"
        "t.manual_seed(0)\n"
        "block = ResBlock(channels=4)\n"
        "assert hasattr(block, 'main'), \"block must expose its main path as self.main\"\n"
        "assert isinstance(block.main, nn.Sequential), f'self.main must be nn.Sequential, got {type(block.main).__name__}'\n"
        "layers = list(block.main)\n"
        "assert len(layers) == 5, f'expected 5 layers in main, got {len(layers)}'\n"
        "assert isinstance(layers[0], nn.Conv2d)\n"
        "assert isinstance(layers[1], nn.BatchNorm2d)\n"
        "assert isinstance(layers[2], nn.ReLU)\n"
        "assert isinstance(layers[3], nn.Conv2d)\n"
        "assert isinstance(layers[4], nn.BatchNorm2d)\n"
        "# Crucially: no final ReLU inside main — must be exactly 5.\n"
        "\n"
        "# Case B: shape contract.\n"
        "block.eval()  # so BN uses running stats (which are 0/1 by default) deterministically.\n"
        "x = t.randn(2, 4, 6, 8)\n"
        "out = block(x)\n"
        "assert tuple(out.shape) == (2, 4, 6, 8)\n"
        "\n"
        "# Case C: residual-add is load-bearing — zero out the conv weights and BN biases.\n"
        "with t.no_grad():\n"
        "    for m in block.main:\n"
        "        if isinstance(m, nn.Conv2d):\n"
        "            m.weight.zero_()\n"
        "        if isinstance(m, nn.BatchNorm2d):\n"
        "            m.weight.fill_(1.0)\n"
        "            m.bias.zero_()\n"
        "out = block(x)\n"
        "# With all convs zero, main(x) == 0 in eval, so forward returns relu(0 + x) = relu(x).\n"
        "assert t.allclose(out, F.relu(x), atol=1e-5), (\n"
        "    'zeroed-conv block must return relu(x) — skip-add missing or applied to wrong tensor'\n"
        ")\n"
        "\n"
        "# Case D: parameters of self.main are reachable from block.parameters() (composition wires fan-in).\n"
        "block2 = ResBlock(channels=3)\n"
        "block_params = {id(p) for p in block2.parameters()}\n"
        "main_params = {id(p) for p in block2.main.parameters()}\n"
        "assert main_params.issubset(block_params), (\n"
        "    'parameters of self.main must propagate to block.parameters() — composition broken'\n"
        ")"
    ),
    "solution_body": (
        "def cx26_make_resblock_with_sequential():\n"
        "    class ResBlock(nn.Module):\n"
        "        def __init__(self, channels):\n"
        "            super().__init__()\n"
        "            # Atom B (module-composition): the linear stretch lives inside an nn.Sequential.\n"
        "            # NOTE: no final ReLU inside self.main — that's in forward AFTER the skip-add.\n"
        "            self.main = nn.Sequential(\n"
        "                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),\n"
        "                nn.BatchNorm2d(channels),\n"
        "                nn.ReLU(),\n"
        "                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),\n"
        "                nn.BatchNorm2d(channels),\n"
        "            )\n"
        "\n"
        "        def forward(self, x):\n"
        "            # Atom A (residual-skip-add): main path + skip, then final ReLU.\n"
        "            return F.relu(self.main(x) + x)\n"
        "\n"
        "    return ResBlock"
    ),
    "solution_notes": (
        "The two atoms split the responsibility cleanly: `self.main` (composition) is the "
        "architecture knob — swap it for a bottleneck variant and everything else still works. "
        "The skip-add lives in `forward` because the outer module is the only place where the "
        "input `x` and the main-path output `self.main(x)` co-exist."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["residual-skip-add", "module-composition"],
    "lo": (
        "Compose nn.Sequential (linear stretch inside the block) with residual-skip-add "
        "(input added to the Sequential's output before the final ReLU) to express the canonical "
        "ResNet block as composition of two atoms."
    ),
}


# ===========================================================================
# cx27 — stride-2 residual block: main path downsamples, skip must match
# ===========================================================================
spec_27 = {
    "atom_ids": ["conv-stride-downsample", "residual-skip-add"],
    "subtopics": _subs(["conv-stride-downsample", "residual-skip-add"]),
    "primary_atom": "residual-skip-add",
    "part": "part2",
    "exercise_index": 27,
    "exercise_title": "stride-2 conv with a matching 1x1 skip downsample",
    "slug": "stride2-conv-with-matching-skip-downsample",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "When a ResNet block **downsamples** (output spatial size = input // 2) the main path uses "
        "a stride-2 conv. But then the **skip can't be a plain identity any more** — the input "
        "is `(N, C_in, H, W)` and the main-path output is `(N, C_out, H/2, W/2)`. They don't add.\n\n"
        "The fix: project the skip through a **1x1 conv with stride 2** that matches both the "
        "channel change AND the spatial change. The shape arithmetic for stride-2:\n"
        "- `H_out = floor((H + 2*pad - kernel) / stride) + 1`\n"
        "- For `k=3, pad=1, stride=2`: `H_out = floor((H + 2 - 3) / 2) + 1 = floor((H - 1) / 2) + 1 = ceil(H/2)`.\n"
        "- For `k=1, pad=0, stride=2`: `H_out = floor((H - 1) / 2) + 1 = ceil(H/2)`. SAME formula.\n\n"
        "Both convs land on the same `(H_out, W_out)`, so the add type-checks. Channels must also "
        "match: skip projection is `Conv2d(C_in, C_out, kernel_size=1, stride=2, bias=False)`.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "class DownBlock(nn.Module):\n"
        "    def __init__(self, c_in, c_out):\n"
        "        super().__init__()\n"
        "        self.main = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)\n"
        "        self.skip = nn.Conv2d(c_in, c_out, 1, stride=2, bias=False)\n"
        "    def forward(self, x):\n"
        "        return F.relu(self.main(x) + self.skip(x))\n"
        "```\n\n"
        "**Why care.** ARENA's `BlockGroup` does exactly this at the start of each stage. Getting "
        "the skip downsample wrong is the most common ResNet shape mismatch."
    ),
    "prompt_body": (
        "Implement `cx27_make_downsample_block()` — return the class `DownBlock(nn.Module)`.\n\n"
        "Required structure:\n"
        "- `__init__(self, c_in, c_out)`:\n"
        "  - `super().__init__()`\n"
        "  - `self.main = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False)`\n"
        "  - `self.skip = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, padding=0, bias=False)`\n"
        "- `forward(self, x)` returns `F.relu(self.main(x) + self.skip(x))`.\n\n"
        "Spatial arithmetic: for input `(N, c_in, H, W)`, both `self.main(x)` and `self.skip(x)` "
        "produce `(N, c_out, ceil(H/2), ceil(W/2))`. They MUST line up — that's the whole point.\n\n"
        "The test fuzzes several `(c_in, c_out, H, W)` combos and asserts: (1) output shape matches "
        "the stride-2 formula, (2) the skip is actually being added (zeroing the main's weight "
        "leaves the skip projection's output, NOT zero)."
    ),
    "stub_body": (
        "def cx27_make_downsample_block():\n"
        "    \"\"\"Return the DownBlock class.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "DownBlock = cx27_make_downsample_block()\n"
        "assert issubclass(DownBlock, nn.Module)\n"
        "\n"
        "# Case A: instantiate and inspect the two child convs.\n"
        "t.manual_seed(0)\n"
        "block = DownBlock(c_in=4, c_out=8)\n"
        "assert isinstance(block.main, nn.Conv2d) and isinstance(block.skip, nn.Conv2d)\n"
        "assert block.main.stride == (2, 2), f'main must have stride=2, got {block.main.stride}'\n"
        "assert block.skip.stride == (2, 2), f'skip must have stride=2 to match, got {block.skip.stride}'\n"
        "assert block.main.kernel_size == (3, 3)\n"
        "assert block.skip.kernel_size == (1, 1), f'skip must be 1x1, got {block.skip.kernel_size}'\n"
        "assert block.main.in_channels == 4 and block.main.out_channels == 8\n"
        "assert block.skip.in_channels == 4 and block.skip.out_channels == 8\n"
        "\n"
        "# Case B: shape contract — stride-2 formula must match for several spatial sizes.\n"
        "import math\n"
        "for c_in, c_out, H, W in [(4, 8, 8, 8), (3, 6, 5, 7), (4, 8, 13, 11), (2, 4, 16, 16)]:\n"
        "    blk = DownBlock(c_in, c_out)\n"
        "    x = t.randn(2, c_in, H, W)\n"
        "    out = blk(x)\n"
        "    # k=3, pad=1, stride=2 and k=1, pad=0, stride=2 BOTH give ceil(H/2), ceil(W/2).\n"
        "    expected_H = math.ceil(H / 2)\n"
        "    expected_W = math.ceil(W / 2)\n"
        "    assert tuple(out.shape) == (2, c_out, expected_H, expected_W), (\n"
        "        f'cin={c_in} cout={c_out} H={H} W={W}: expected '\n"
        "        f'(2,{c_out},{expected_H},{expected_W}), got {tuple(out.shape)}'\n"
        "    )\n"
        "\n"
        "# Case C: residual-add is load-bearing — zero main, output must equal relu(skip(x)).\n"
        "block = DownBlock(c_in=3, c_out=5)\n"
        "x = t.randn(2, 3, 6, 6)\n"
        "with t.no_grad():\n"
        "    block.main.weight.zero_()\n"
        "out = block(x)\n"
        "expected = F.relu(block.skip(x))\n"
        "assert t.allclose(out, expected, atol=1e-6), 'zeroed main: output should equal relu(skip(x))'\n"
        "# Also: out should NOT be all zero (skip is nonzero) — proves skip was added, not dropped.\n"
        "assert out.abs().sum().item() > 0, 'out all-zero — skip projection missing'\n"
        "\n"
        "# Case D: zero the SKIP — output should equal relu(main(x)).\n"
        "block = DownBlock(c_in=3, c_out=5)\n"
        "x = t.randn(2, 3, 6, 6)\n"
        "with t.no_grad():\n"
        "    block.skip.weight.zero_()\n"
        "out = block(x)\n"
        "expected = F.relu(block.main(x))\n"
        "assert t.allclose(out, expected, atol=1e-6), 'zeroed skip: output should equal relu(main(x))'"
    ),
    "solution_body": (
        "def cx27_make_downsample_block():\n"
        "    class DownBlock(nn.Module):\n"
        "        def __init__(self, c_in, c_out):\n"
        "            super().__init__()\n"
        "            # Atom A (conv-stride-downsample): main 3x3 stride-2 halves H,W (ceil rule).\n"
        "            self.main = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False)\n"
        "            # Skip MUST land on the same (H/2, W/2). 1x1 stride-2 gives ceil(H/2), ceil(W/2).\n"
        "            self.skip = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, padding=0, bias=False)\n"
        "\n"
        "        def forward(self, x):\n"
        "            # Atom B (residual-skip-add): add the (matched-shape) projections.\n"
        "            return F.relu(self.main(x) + self.skip(x))\n"
        "\n"
        "    return DownBlock"
    ),
    "solution_notes": (
        "The trick is that two SEEMINGLY DIFFERENT conv configs (`k=3 p=1 s=2` and `k=1 p=0 s=2`) "
        "produce the same `ceil(H/2)` output spatial size. That's not a coincidence — it's the "
        "PyTorch stride formula `floor((H + 2p - k)/s) + 1` evaluating to the same expression for "
        "both. Drop the `padding=1` on main or change the skip kernel and the shapes diverge."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["conv-stride-downsample", "residual-skip-add"],
    "lo": (
        "Compose stride-2 conv downsampling (3x3 main with padding=1) with residual-skip-add "
        "(1x1 stride-2 skip projection that matches both channels and spatial halving) to express "
        "the canonical ARENA BlockGroup-entry block."
    ),
}


# ===========================================================================
# cx28 — full BatchNorm forward: affine + running stats
# ===========================================================================
spec_28 = {
    "atom_ids": ["batchnorm-affine-params", "batchnorm-running-stats"],
    "subtopics": _subs(["batchnorm-affine-params", "batchnorm-running-stats"]),
    "primary_atom": "batchnorm-affine-params",
    "part": "part2",
    "exercise_index": 28,
    "exercise_title": "BatchNorm2d forward with both affine params and running stats",
    "slug": "batchnorm-affine-and-running-stats",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "A BatchNorm layer carries TWO kinds of state:\n"
        "1. **Affine params** — `gamma` (a.k.a. `weight`) and `beta` (`bias`), both shape `(C,)`. "
        "These are LEARNED via SGD. They scale and shift the normalised activations: "
        "`y = gamma * (x - mu) / sqrt(var + eps) + beta`. They are wrapped in `nn.Parameter` so "
        "the optimizer sees them.\n"
        "2. **Running stats** — `running_mean` and `running_var`, also shape `(C,)`. These are "
        "NOT parameters — they are *buffers*, updated by a moving average over training batches "
        "and used unchanged at inference. They must be registered with `self.register_buffer(...)` "
        "so `.to(device)` moves them and `.state_dict()` saves them, but the optimizer does NOT "
        "see them.\n\n"
        "**Anatomy of a manual BN forward (train mode).**\n"
        "```python\n"
        "mu = x.mean(dim=(0, 2, 3))                     # (C,) per-channel batch mean.\n"
        "var = x.var(dim=(0, 2, 3), unbiased=False)     # (C,) per-channel batch var.\n"
        "# Update running stats in-place with momentum.\n"
        "self.running_mean.mul_(1 - momentum).add_(mu * momentum)\n"
        "self.running_var.mul_(1 - momentum).add_(var * momentum)\n"
        "# Normalize using BATCH stats during training.\n"
        "x_hat = (x - mu[None, :, None, None]) / t.sqrt(var[None, :, None, None] + eps)\n"
        "# Apply learned affine — gamma * x_hat + beta.\n"
        "return self.weight[None, :, None, None] * x_hat + self.bias[None, :, None, None]\n"
        "```\n\n"
        "**Why both atoms together.** Affine without running stats = a normalised activation with "
        "no inference-time stand-in for the batch mean/var. Running stats without affine = no "
        "way for the model to learn to undo the normalisation when that's what helps. They MUST "
        "coexist; one is `Parameter`, the other is `buffer`."
    ),
    "prompt_body": (
        "Implement `cx28_make_batchnorm2d()` — return the class `MyBatchNorm2d(nn.Module)`.\n\n"
        "Required structure:\n"
        "- `__init__(self, num_features, eps=1e-5, momentum=0.1)`:\n"
        "  - `super().__init__()`\n"
        "  - `self.eps = eps; self.momentum = momentum`\n"
        "  - Affine params (atom: batchnorm-affine-params):\n"
        "    - `self.weight = nn.Parameter(t.ones(num_features))`\n"
        "    - `self.bias   = nn.Parameter(t.zeros(num_features))`\n"
        "  - Running stats (atom: batchnorm-running-stats — REGISTER as BUFFERS):\n"
        "    - `self.register_buffer('running_mean', t.zeros(num_features))`\n"
        "    - `self.register_buffer('running_var',  t.ones(num_features))`\n"
        "- `forward(self, x)` — assume `self.training is True` (we test eval-vs-train in cx29):\n"
        "  - Compute per-channel `mu`, `var` over `(N, H, W)` (i.e. `dim=(0, 2, 3)`, `unbiased=False`).\n"
        "  - Update `self.running_mean` and `self.running_var` in-place with momentum (formula: "
        "`new = (1 - m) * old + m * batch`).\n"
        "  - Normalize x using the BATCH stats, apply affine. Return the result.\n\n"
        "The test checks: (a) `weight` and `bias` are Parameters; (b) `running_mean` and "
        "`running_var` are in `state_dict` but NOT in `parameters()`; (c) the forward output equals "
        "`F.batch_norm(x, running_mean, running_var, weight, bias, training=True, momentum=...)`; "
        "(d) after the forward, `running_mean` and `running_var` were updated by the momentum rule."
    ),
    "stub_body": (
        "def cx28_make_batchnorm2d():\n"
        "    \"\"\"Return the MyBatchNorm2d class.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "MyBN = cx28_make_batchnorm2d()\n"
        "assert issubclass(MyBN, nn.Module)\n"
        "\n"
        "# Case A: affine params are Parameters; running stats are buffers (NOT parameters).\n"
        "t.manual_seed(0)\n"
        "bn = MyBN(num_features=4)\n"
        "param_names = {name for name, _ in bn.named_parameters()}\n"
        "buffer_names = {name for name, _ in bn.named_buffers()}\n"
        "assert 'weight' in param_names and 'bias' in param_names, (\n"
        "    f'weight & bias must be nn.Parameter; param names = {param_names}'\n"
        ")\n"
        "assert 'running_mean' in buffer_names and 'running_var' in buffer_names, (\n"
        "    f'running_mean & running_var must be registered buffers; buffer names = {buffer_names}'\n"
        ")\n"
        "# Running stats must NOT appear in parameters — that would let SGD overwrite them.\n"
        "assert 'running_mean' not in param_names, 'running_mean is a buffer, not a Parameter'\n"
        "assert 'running_var' not in param_names, 'running_var is a buffer, not a Parameter'\n"
        "\n"
        "# Case B: initial values — weight=1, bias=0, running_mean=0, running_var=1.\n"
        "assert t.allclose(bn.weight, t.ones(4))\n"
        "assert t.allclose(bn.bias, t.zeros(4))\n"
        "assert t.allclose(bn.running_mean, t.zeros(4))\n"
        "assert t.allclose(bn.running_var, t.ones(4))\n"
        "\n"
        "# Case C: forward output equals F.batch_norm reference.\n"
        "t.manual_seed(1)\n"
        "bn = MyBN(num_features=3, eps=1e-5, momentum=0.1)\n"
        "# Make affine params non-trivial so the affine half actually matters.\n"
        "with t.no_grad():\n"
        "    bn.weight.copy_(t.tensor([0.5, 2.0, 1.5]))\n"
        "    bn.bias.copy_(t.tensor([0.1, -0.2, 0.3]))\n"
        "    bn.running_mean.copy_(t.tensor([0.0, 0.0, 0.0]))\n"
        "    bn.running_var.copy_(t.tensor([1.0, 1.0, 1.0]))\n"
        "# Save copies of running stats BEFORE the forward (which mutates them).\n"
        "rm_before = bn.running_mean.clone()\n"
        "rv_before = bn.running_var.clone()\n"
        "x = t.randn(2, 3, 4, 5)\n"
        "# Reference computed with F.batch_norm (training=True so it updates the stats we pass).\n"
        "rm_ref = rm_before.clone()\n"
        "rv_ref = rv_before.clone()\n"
        "ref = F.batch_norm(x, rm_ref, rv_ref, bn.weight, bn.bias, training=True, momentum=0.1, eps=1e-5)\n"
        "out = bn(x)\n"
        "assert tuple(out.shape) == (2, 3, 4, 5)\n"
        "assert t.allclose(out, ref, atol=1e-5), (\n"
        "    f'forward output mismatch with F.batch_norm reference; max err = {(out - ref).abs().max().item()}'\n"
        ")\n"
        "\n"
        "# Case D: running stats were updated by the momentum rule.\n"
        "assert not t.allclose(bn.running_mean, rm_before), 'running_mean must update during train forward'\n"
        "assert not t.allclose(bn.running_var, rv_before), 'running_var must update during train forward'\n"
        "# Cross-check the exact updated values match the reference.\n"
        "assert t.allclose(bn.running_mean, rm_ref, atol=1e-5), 'running_mean update rule mismatch'\n"
        "assert t.allclose(bn.running_var, rv_ref, atol=1e-5), 'running_var update rule mismatch'"
    ),
    "solution_body": (
        "def cx28_make_batchnorm2d():\n"
        "    class MyBatchNorm2d(nn.Module):\n"
        "        def __init__(self, num_features, eps=1e-5, momentum=0.1):\n"
        "            super().__init__()\n"
        "            self.eps = eps\n"
        "            self.momentum = momentum\n"
        "            # Atom A (batchnorm-affine-params): learned scale gamma and shift beta.\n"
        "            self.weight = nn.Parameter(t.ones(num_features))\n"
        "            self.bias = nn.Parameter(t.zeros(num_features))\n"
        "            # Atom B (batchnorm-running-stats): tracked via register_buffer — moves with .to(),\n"
        "            # saved by state_dict(), but the optimizer doesn't update them.\n"
        "            self.register_buffer('running_mean', t.zeros(num_features))\n"
        "            self.register_buffer('running_var', t.ones(num_features))\n"
        "\n"
        "        def forward(self, x):\n"
        "            # Per-channel batch statistics (reduce N, H, W; keep C).\n"
        "            mu = x.mean(dim=(0, 2, 3))\n"
        "            var = x.var(dim=(0, 2, 3), unbiased=False)\n"
        "            # Momentum-update the buffers in-place.\n"
        "            self.running_mean.mul_(1 - self.momentum).add_(mu.detach() * self.momentum)\n"
        "            # Note: PyTorch uses UNBIASED var for the running stat update.\n"
        "            n = x.shape[0] * x.shape[2] * x.shape[3]\n"
        "            var_unbiased = var * (n / (n - 1)) if n > 1 else var\n"
        "            self.running_var.mul_(1 - self.momentum).add_(var_unbiased.detach() * self.momentum)\n"
        "            # Normalize with BATCH stats during training, then affine.\n"
        "            x_hat = (x - mu[None, :, None, None]) / t.sqrt(var[None, :, None, None] + self.eps)\n"
        "            return self.weight[None, :, None, None] * x_hat + self.bias[None, :, None, None]\n"
        "\n"
        "    return MyBatchNorm2d"
    ),
    "solution_notes": (
        "Two subtleties: (1) the batch var used for NORMALIZATION is biased (`unbiased=False`), "
        "but the var used to update `running_var` is UNBIASED — PyTorch's actual behaviour. "
        "(2) `register_buffer` is the canonical mechanism for non-trainable state on a Module. "
        "It's not just a `self.x = tensor` — registering puts the tensor in `state_dict()` "
        "and `to(device)`'s sweep, both of which `nn.Parameter` would also do, but with the "
        "crucial difference that buffers are skipped by the optimizer."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["batchnorm-affine-params", "batchnorm-running-stats"],
    "lo": (
        "Compose the BatchNorm affine params (gamma/beta as nn.Parameter) with running stats "
        "(running_mean/running_var via register_buffer) to implement a full training-mode "
        "BatchNorm2d forward that matches F.batch_norm."
    ),
}


# ===========================================================================
# cx29 — eval mode uses running stats; train mode uses batch stats
# ===========================================================================
spec_29 = {
    "atom_ids": ["inference-mode-step", "train-eval-mode-branch"],
    "subtopics": _subs(["inference-mode-step", "train-eval-mode-branch"]),
    "primary_atom": "train-eval-mode-branch",
    "part": "part2",
    "exercise_index": 29,
    "exercise_title": "BatchNorm2d that branches on self.training (eval uses running stats)",
    "slug": "bn-train-eval-mode-branch",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "BatchNorm has two completely different behaviours, switched by `self.training`:\n"
        "- **Train mode** (`self.training is True`): normalise with the CURRENT BATCH's mean/var, "
        "and update the running stats by exponential moving average. Gradients flow through "
        "the normalisation.\n"
        "- **Eval / inference mode** (`self.training is False`): normalise with the FROZEN "
        "`running_mean` / `running_var`. Do NOT update them. This is what gives BN a deterministic "
        "test-time function, independent of batch composition.\n\n"
        "**The two atoms.**\n"
        "- **train-eval-mode-branch** — the `if self.training: ... else: ...` switch inside "
        "`forward`. Without this branch, a batch of 1 image at test time gets `var = 0` and the "
        "output is NaN.\n"
        "- **inference-mode-step** — the eval branch. It wraps the forward in `t.no_grad()` or "
        "uses `t.inference_mode()` to avoid building autograd machinery the inference path doesn't "
        "need. (`.eval()` flips `self.training`; `t.inference_mode()` is the autograd-side switch.)\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "def forward(self, x):\n"
        "    if self.training:\n"
        "        mu = x.mean(dim=(0, 2, 3)); var = x.var(dim=(0, 2, 3), unbiased=False)\n"
        "        # ... update running stats ...\n"
        "    else:\n"
        "        # inference-mode-step: use frozen buffers, no autograd.\n"
        "        mu = self.running_mean; var = self.running_var\n"
        "    x_hat = (x - mu[None,:,None,None]) / t.sqrt(var[None,:,None,None] + eps)\n"
        "    return gamma * x_hat + beta\n"
        "```\n\n"
        "**Why care.** Forgetting `model.eval()` before validation is one of the most common "
        "ResNet bugs. Test loss looks worse than train loss because BN is still using each "
        "validation batch's stats (often differently distributed)."
    ),
    "prompt_body": (
        "Implement `cx29_make_bn_with_branch()` — return `MyBN(nn.Module)` with the train/eval "
        "branch wired in.\n\n"
        "Required structure (extends cx28):\n"
        "- `__init__(self, num_features, eps=1e-5, momentum=0.1)` — same as cx28: `weight`, "
        "`bias` as Parameters; `running_mean`, `running_var` as buffers.\n"
        "- `forward(self, x)` — **branch on `self.training`**:\n"
        "  - If training: compute batch stats, UPDATE running stats, normalise with BATCH stats.\n"
        "  - If eval: normalise with `self.running_mean` / `self.running_var`, do NOT update.\n"
        "  - In BOTH branches: apply affine (`gamma * x_hat + beta`).\n\n"
        "The test:\n"
        "- Sets `bn.training = True`, runs forward, records `running_mean` / `running_var` changes.\n"
        "- Sets `bn.training = False`, runs forward AGAIN — running stats MUST NOT change, and "
        "the output MUST equal `F.batch_norm(..., training=False, ...)`.\n"
        "- Sanity: train-mode output for a uniform-mean batch != eval-mode output (different mu/var).\n"
        "- The eval branch is also tested under `t.no_grad()` to confirm it works in inference mode."
    ),
    "stub_body": (
        "def cx29_make_bn_with_branch():\n"
        "    \"\"\"Return the MyBN class with train/eval branch.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "MyBN = cx29_make_bn_with_branch()\n"
        "assert issubclass(MyBN, nn.Module)\n"
        "\n"
        "t.manual_seed(0)\n"
        "bn = MyBN(num_features=3, eps=1e-5, momentum=0.1)\n"
        "# Push running stats away from defaults so we can tell train- vs eval-mode apart.\n"
        "with t.no_grad():\n"
        "    bn.running_mean.copy_(t.tensor([0.5, -0.5, 1.0]))\n"
        "    bn.running_var.copy_(t.tensor([2.0, 0.5, 1.5]))\n"
        "    bn.weight.copy_(t.tensor([1.2, 0.8, 1.5]))\n"
        "    bn.bias.copy_(t.tensor([-0.1, 0.2, 0.3]))\n"
        "\n"
        "x = t.randn(4, 3, 5, 6) + 3.0  # nonzero mean — so batch vs running stats differ.\n"
        "\n"
        "# Case A: TRAIN mode — output matches F.batch_norm(training=True).\n"
        "bn.train()  # sets self.training = True.\n"
        "assert bn.training is True\n"
        "rm_before = bn.running_mean.clone()\n"
        "rv_before = bn.running_var.clone()\n"
        "rm_ref = rm_before.clone()\n"
        "rv_ref = rv_before.clone()\n"
        "ref_train = F.batch_norm(x, rm_ref, rv_ref, bn.weight, bn.bias, training=True, momentum=0.1, eps=1e-5)\n"
        "out_train = bn(x)\n"
        "assert t.allclose(out_train, ref_train, atol=1e-5), 'train-mode output != F.batch_norm reference'\n"
        "# Running stats moved.\n"
        "assert not t.allclose(bn.running_mean, rm_before), 'train mode must update running_mean'\n"
        "\n"
        "# Case B: EVAL mode — output uses running stats; stats DO NOT update.\n"
        "bn.eval()  # sets self.training = False.\n"
        "assert bn.training is False\n"
        "rm_before_eval = bn.running_mean.clone()\n"
        "rv_before_eval = bn.running_var.clone()\n"
        "ref_eval = F.batch_norm(\n"
        "    x, bn.running_mean, bn.running_var, bn.weight, bn.bias,\n"
        "    training=False, eps=1e-5,\n"
        ")\n"
        "out_eval = bn(x)\n"
        "assert t.allclose(out_eval, ref_eval, atol=1e-5), 'eval-mode output != F.batch_norm(training=False) reference'\n"
        "assert t.allclose(bn.running_mean, rm_before_eval), 'eval mode must NOT update running_mean'\n"
        "assert t.allclose(bn.running_var, rv_before_eval), 'eval mode must NOT update running_var'\n"
        "\n"
        "# Case C: train and eval outputs differ (proves the branch matters).\n"
        "assert not t.allclose(out_train, out_eval), (\n"
        "    'train- and eval-mode outputs are identical — the branch on self.training is missing'\n"
        ")\n"
        "\n"
        "# Case D: inference-mode-step — eval path must work under t.no_grad().\n"
        "bn.eval()\n"
        "rm_before_ng = bn.running_mean.clone()\n"
        "with t.no_grad():\n"
        "    out_ng = bn(x)\n"
        "# Same output as Case B.\n"
        "assert t.allclose(out_ng, ref_eval, atol=1e-5)\n"
        "# Output should have no grad_fn (we were under no_grad).\n"
        "assert out_ng.requires_grad is False, 'eval forward under t.no_grad() should not require grad'\n"
        "# And running stats still untouched.\n"
        "assert t.allclose(bn.running_mean, rm_before_ng), 'inference-mode forward must not touch running stats'"
    ),
    "solution_body": (
        "def cx29_make_bn_with_branch():\n"
        "    class MyBN(nn.Module):\n"
        "        def __init__(self, num_features, eps=1e-5, momentum=0.1):\n"
        "            super().__init__()\n"
        "            self.eps = eps\n"
        "            self.momentum = momentum\n"
        "            self.weight = nn.Parameter(t.ones(num_features))\n"
        "            self.bias = nn.Parameter(t.zeros(num_features))\n"
        "            self.register_buffer('running_mean', t.zeros(num_features))\n"
        "            self.register_buffer('running_var', t.ones(num_features))\n"
        "\n"
        "        def forward(self, x):\n"
        "            # Atom A (train-eval-mode-branch): self.training switches the stat source.\n"
        "            if self.training:\n"
        "                mu = x.mean(dim=(0, 2, 3))\n"
        "                var_biased = x.var(dim=(0, 2, 3), unbiased=False)\n"
        "                # Update running stats with the unbiased var (PyTorch convention).\n"
        "                n = x.shape[0] * x.shape[2] * x.shape[3]\n"
        "                var_unbiased = var_biased * (n / (n - 1)) if n > 1 else var_biased\n"
        "                self.running_mean.mul_(1 - self.momentum).add_(mu.detach() * self.momentum)\n"
        "                self.running_var.mul_(1 - self.momentum).add_(var_unbiased.detach() * self.momentum)\n"
        "                mu_used, var_used = mu, var_biased\n"
        "            else:\n"
        "                # Atom B (inference-mode-step): frozen running stats; no update; works under no_grad.\n"
        "                mu_used, var_used = self.running_mean, self.running_var\n"
        "            x_hat = (x - mu_used[None, :, None, None]) / t.sqrt(var_used[None, :, None, None] + self.eps)\n"
        "            return self.weight[None, :, None, None] * x_hat + self.bias[None, :, None, None]\n"
        "\n"
        "    return MyBN"
    ),
    "solution_notes": (
        "Two flags often get confused: `self.training` (Module-level, flipped by `.train()/.eval()`) "
        "and `t.is_grad_enabled()` (global, flipped by `t.no_grad()` / `t.inference_mode()`). The "
        "BN branch above only looks at `self.training` — autograd-disabled inference is orthogonal "
        "and works for free because we never touch a Parameter that needs grad on the eval path. "
        "The `.detach()` on `mu` and `var_unbiased` when updating buffers prevents the running-stat "
        "update from accidentally building a gradient back through the buffer."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["inference-mode-step", "train-eval-mode-branch"],
    "lo": (
        "Compose the train/eval-mode branch on self.training (batch stats + update vs running "
        "stats + no update) with the inference-mode step (eval branch works correctly under "
        "t.no_grad()) to implement BN's dual behaviour."
    ),
}


# ===========================================================================
# cx30 — Kaiming init + nn.Parameter wrap
# ===========================================================================
spec_30 = {
    "atom_ids": ["kaiming-uniform-sf-init", "nn-parameter-wrap"],
    "subtopics": _subs(["kaiming-uniform-sf-init", "nn-parameter-wrap"]),
    "primary_atom": "kaiming-uniform-sf-init",
    "part": "part2",
    "exercise_index": 30,
    "exercise_title": "Kaiming-uniform initialised weight, wrapped as nn.Parameter",
    "slug": "kaiming-uniform-init-as-parameter",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "PyTorch's `nn.Linear` and `nn.Conv2d` both initialise their weight tensor via **Kaiming "
        "uniform** (scaled by `1/sqrt(fan_in)` with a gain matching the assumed downstream "
        "nonlinearity). When you write a custom layer, you have to do this yourself: create a "
        "tensor of the right shape, fill it with kaiming-uniform samples, then **wrap it as "
        "`nn.Parameter`** so the optimizer sees it.\n\n"
        "**The two atoms.**\n"
        "- **kaiming-uniform-sf-init** — the FILL. For a weight of shape `(out, in)` (Linear) or "
        "`(out_c, in_c, kH, kW)` (Conv2d), `fan_in = in` (Linear) or `in_c * kH * kW` (Conv2d). "
        "The 'self-fan' style ARENA uses is "
        "`bound = 1 / sqrt(fan_in); weight.uniform_(-bound, bound)`. (`nn.init.kaiming_uniform_` "
        "with `a=sqrt(5)` matches PyTorch's exact convention for Linear; for clarity ARENA writes "
        "the formula by hand.)\n"
        "- **nn-parameter-wrap** — `nn.Parameter(tensor)` flags the tensor as 'this is a learned "
        "param'. After wrapping, the tensor appears in `model.parameters()` and the optimizer's "
        "update rule applies to it. A raw tensor in `self.x = tensor` is NOT trained.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "class MyLinear(nn.Module):\n"
        "    def __init__(self, in_f, out_f):\n"
        "        super().__init__()\n"
        "        # Atom A: kaiming-uniform-sf-init.\n"
        "        bound = 1.0 / math.sqrt(in_f)\n"
        "        w = t.empty(out_f, in_f).uniform_(-bound, bound)\n"
        "        # Atom B: nn-parameter-wrap.\n"
        "        self.weight = nn.Parameter(w)\n"
        "    def forward(self, x):\n"
        "        return x @ self.weight.T\n"
        "```\n\n"
        "**Why both atoms together.** A correctly-shaped weight tensor is useless if the optimizer "
        "can't find it; a correctly-wrapped `nn.Parameter` is broken if its init scale is wrong "
        "(too big → activations explode; too small → vanishing grads on first batch)."
    ),
    "prompt_body": (
        "Implement two helpers.\n\n"
        "1. `cx30_init_kaiming_uniform(shape, fan_in)` — return a fresh `t.Tensor` of the given "
        "`shape`, filled with samples from `Uniform(-bound, +bound)` where `bound = 1 / sqrt(fan_in)`. "
        "Do NOT wrap as Parameter — just return the raw tensor.\n\n"
        "2. `cx30_make_linear()` — return the class `MyLinear(nn.Module)` such that "
        "`MyLinear(in_features, out_features)`:\n"
        "   - calls `super().__init__()`\n"
        "   - creates `weight` via `cx30_init_kaiming_uniform((out_features, in_features), in_features)`\n"
        "   - **wraps it as `nn.Parameter`** and stores as `self.weight`\n"
        "   - implements `forward(self, x)` as `x @ self.weight.T`\n\n"
        "The test checks: (a) the helper produces a tensor in the right `[-bound, +bound]` range "
        "and stats consistent with a uniform; (b) `MyLinear.weight` is an `nn.Parameter` "
        "(not just a Tensor) and shows up in `.parameters()`; (c) gradients flow through "
        "`self.weight` after one backward pass; (d) the init scale obeys `1/sqrt(fan_in)` for "
        "different `in_features`."
    ),
    "stub_body": (
        "def cx30_init_kaiming_uniform(shape, fan_in):\n"
        "    \"\"\"Return a tensor of `shape` filled with U(-1/sqrt(fan_in), +1/sqrt(fan_in)).\"\"\"\n"
        "    raise NotImplementedError\n"
        "\n"
        "def cx30_make_linear():\n"
        "    \"\"\"Return the MyLinear class.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "import math\n"
        "\n"
        "# Case A: helper produces correctly-shaped tensor with correct bound.\n"
        "t.manual_seed(0)\n"
        "w = cx30_init_kaiming_uniform((20, 100), fan_in=100)\n"
        "assert isinstance(w, t.Tensor), f'helper must return a Tensor, got {type(w).__name__}'\n"
        "assert tuple(w.shape) == (20, 100)\n"
        "bound = 1.0 / math.sqrt(100)  # = 0.1\n"
        "assert w.min().item() >= -bound - 1e-6 and w.max().item() <= bound + 1e-6, (\n"
        "    f'values must lie in [-{bound}, +{bound}]; got [{w.min().item()}, {w.max().item()}]'\n"
        ")\n"
        "# Statistical sanity: uniform [-bound, +bound] has var = bound^2 / 3.\n"
        "expected_var = bound ** 2 / 3.0\n"
        "observed_var = w.var().item()\n"
        "assert abs(observed_var - expected_var) / expected_var < 0.1, (\n"
        "    f'variance {observed_var:.5f} is far from uniform-bound expected {expected_var:.5f}'\n"
        ")\n"
        "\n"
        "# Case B: scale changes with fan_in.\n"
        "t.manual_seed(1)\n"
        "w_small = cx30_init_kaiming_uniform((50, 4),   fan_in=4)\n"
        "w_large = cx30_init_kaiming_uniform((50, 400), fan_in=400)\n"
        "# Smaller fan_in -> larger bound -> larger std.\n"
        "assert w_small.std().item() > w_large.std().item() * 5, (\n"
        "    'std with fan_in=4 should be much larger than with fan_in=400 — scale rule broken'\n"
        ")\n"
        "\n"
        "# Case C: MyLinear wraps weight as nn.Parameter.\n"
        "MyLinear = cx30_make_linear()\n"
        "assert issubclass(MyLinear, nn.Module)\n"
        "t.manual_seed(2)\n"
        "lin = MyLinear(8, 16)\n"
        "assert isinstance(lin.weight, nn.Parameter), (\n"
        "    f'lin.weight must be nn.Parameter (not raw Tensor); got {type(lin.weight).__name__}'\n"
        ")\n"
        "params = list(lin.parameters())\n"
        "assert any(p is lin.weight for p in params), 'lin.weight must appear in lin.parameters()'\n"
        "assert tuple(lin.weight.shape) == (16, 8)\n"
        "\n"
        "# Case D: forward + backward — gradient must flow into lin.weight.\n"
        "x = t.randn(4, 8)\n"
        "y = lin(x)\n"
        "assert tuple(y.shape) == (4, 16)\n"
        "loss = y.pow(2).sum()\n"
        "loss.backward()\n"
        "assert lin.weight.grad is not None, 'no grad on lin.weight — was it wrapped as nn.Parameter?'\n"
        "assert lin.weight.grad.abs().sum().item() > 0, 'grad is zero — backward did not reach lin.weight'\n"
        "\n"
        "# Case E: init scale obeys 1/sqrt(in_features) for MyLinear itself.\n"
        "t.manual_seed(3)\n"
        "lin2 = MyLinear(400, 50)\n"
        "lin3 = MyLinear(4,   50)\n"
        "# Same out_features so std is comparable on the same-shape tensors after.\n"
        "assert lin3.weight.std().item() > lin2.weight.std().item() * 5, (\n"
        "    'MyLinear must apply the 1/sqrt(in_features) scale to its self.weight init'\n"
        ")"
    ),
    "solution_body": (
        "import math\n"
        "\n"
        "def cx30_init_kaiming_uniform(shape, fan_in):\n"
        "    # Atom A (kaiming-uniform-sf-init): bound = 1/sqrt(fan_in), then in-place uniform fill.\n"
        "    bound = 1.0 / math.sqrt(fan_in)\n"
        "    return t.empty(*shape).uniform_(-bound, bound)\n"
        "\n"
        "def cx30_make_linear():\n"
        "    class MyLinear(nn.Module):\n"
        "        def __init__(self, in_features, out_features):\n"
        "            super().__init__()\n"
        "            w = cx30_init_kaiming_uniform((out_features, in_features), fan_in=in_features)\n"
        "            # Atom B (nn-parameter-wrap): without this, the optimizer never sees `weight`.\n"
        "            self.weight = nn.Parameter(w)\n"
        "\n"
        "        def forward(self, x):\n"
        "            return x @ self.weight.T\n"
        "\n"
        "    return MyLinear"
    ),
    "solution_notes": (
        "Two failure modes the test catches: (1) returning a raw tensor instead of `nn.Parameter` "
        "— forward still works, but `lin.parameters()` is empty and the optimizer silently fails to "
        "train the weight. (2) Wrong init scale — e.g. `randn` instead of bounded uniform — "
        "produces values an order of magnitude too large for `fan_in=100`, blowing up the first "
        "forward pass through a deep stack. The `1/sqrt(fan_in)` bound is what `nn.Linear` uses "
        "by default, derived from the 'preserve activation variance' argument in He et al. 2015."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["kaiming-uniform-sf-init", "nn-parameter-wrap"],
    "lo": (
        "Compose Kaiming-uniform self-fan init (uniform fill with bound 1/sqrt(fan_in)) with "
        "nn.Parameter wrapping (so the optimizer can find and update the weight) to build a "
        "minimally-correct linear layer init from scratch."
    ),
}


SPECS = [spec_25, spec_26, spec_27, spec_28, spec_29, spec_30]


def main() -> None:
    for spec in SPECS:
        out = emit_composite(spec)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
