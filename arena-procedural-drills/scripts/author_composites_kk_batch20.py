"""Composite drills cx1..cx6 — batch-20 part5 (KK-cell, ARENA VAE/GAN generator/decoder).

Six composite procedural drills exercising 2-atom pairs from ARENA part 5 —
DCGAN generator + VAE decoder structural building blocks.

cx1  convtranspose-bn-activation-block + nn-module-subclass        — DCGAN G block as nn.Module subclass
cx2  convtranspose-bn-activation-block + rearrange-as-sequential-layer
                                                                    — einops.Rearrange in nn.Sequential before convT
cx3  nn-module-subclass + rearrange-as-sequential-layer            — subclass with einops Rearrange member
cx4  bottleneck-latent-projection + convtranspose-bn-activation-block
                                                                    — latent z -> Linear -> reshape -> upsample blocks
cx5  bottleneck-latent-projection + nn-module-subclass             — Linear-to-feature-map projection in nn.Module
cx6  encoder-decoder-symmetric + convtranspose-bn-activation-block — mirror enc Conv2d/BN/LeakyReLU with dec ConvT2d/BN/ReLU
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_composite import emit_composite  # noqa: E402

INV = {a["atom_id"]: a for a in json.load(open("/tmp/drill_atoms.json"))}


def _subs(atom_ids):
    return [INV[a]["subtopic"] for a in atom_ids]


# part5 atoms uniformly need nn / F / einops.layers.torch.
NN_IMPORTS = [
    "import torch.nn as nn",
    "import torch.nn.functional as F",
    "from einops.layers.torch import Rearrange",
]


# ===========================================================================
# cx1 — DCGAN G block as a real nn.Module subclass.
# ===========================================================================
spec_1 = {
    "atom_ids": ["convtranspose-bn-activation-block", "nn-module-subclass"],
    "subtopics": _subs(["convtranspose-bn-activation-block", "nn-module-subclass"]),
    "primary_atom": "convtranspose-bn-activation-block",
    "part": "part5",
    "exercise_index": 1,
    "exercise_title": "DCGAN G block packaged as an nn.Module subclass",
    "slug": "dcgan-g-block-as-nn-module-subclass",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "The DCGAN generator repeats one structural unit over and over: a "
        "`ConvTranspose2d -> BatchNorm2d -> ReLU` triple that doubles spatial size and halves "
        "channels. Two atoms wire together here:\n\n"
        "1. **convtranspose-bn-activation-block** — the layer triple itself. Canonical hyperparams "
        "are `kernel_size=4, stride=2, padding=1, bias=False`. Output size: "
        "`H_out = (H_in - 1) * stride - 2 * padding + kernel = 2 * H_in`. `bias=False` because "
        "the immediately-following `BatchNorm2d` re-centres anyway, and ReLU (not LeakyReLU) is the "
        "generator-side activation per the DCGAN paper.\n"
        "2. **nn-module-subclass** — wrapping the triple as a `class GBlock(nn.Module): ...` "
        "instead of just returning `nn.Sequential(...)`. The subclass discipline is: call "
        "`super().__init__()` first; register sub-Modules by attribute assignment "
        "(`self.convt = ...`); implement `forward(self, x)`; never call `.forward()` "
        "directly (use `block(x)`).\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "class GBlock(nn.Module):\n"
        "    def __init__(self, in_c, out_c):\n"
        "        super().__init__()                                  # nn-module-subclass.\n"
        "        self.convt = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False)\n"
        "        self.bn    = nn.BatchNorm2d(out_c)                  # convtranspose-bn-activation-block.\n"
        "        self.act   = nn.ReLU(inplace=True)\n"
        "    def forward(self, x):\n"
        "        return self.act(self.bn(self.convt(x)))\n"
        "```\n\n"
        "**Why subclass instead of `nn.Sequential`.** Subclassing lets the block hold extra state "
        "(e.g. a flag, a running counter, a custom init method), expose named children "
        "(`block.convt.weight`), and override `extra_repr` for nicer printing. ARENA's generator "
        "uses `nn.Sequential` of these blocks as the OUTER stack, but each block itself is "
        "typically a named subclass once you want to do init or surgery on it."
    ),
    "prompt_body": (
        "Implement `cx1_make_g_block_cls()` — return a `GBlock` class (an `nn.Module` subclass) "
        "with the following contract:\n\n"
        "- `GBlock(in_channels: int, out_channels: int)` — constructor.\n"
        "- Inside `__init__`, call `super().__init__()` FIRST (atom: nn-module-subclass).\n"
        "- Register three children as attributes (NOT in a `nn.Sequential`):\n"
        "  - `self.convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, "
        "stride=2, padding=1, bias=False)`\n"
        "  - `self.bn = nn.BatchNorm2d(out_channels)`\n"
        "  - `self.act = nn.ReLU(inplace=True)`\n"
        "- `forward(self, x)` runs `convt -> bn -> act` in order (atom: "
        "convtranspose-bn-activation-block).\n\n"
        "The test checks:\n"
        "1. Returned value is a CLASS, not an instance.\n"
        "2. Constructed instance is an `nn.Module` (subclass discipline).\n"
        "3. Children are named `convt`, `bn`, `act` and are of the right types.\n"
        "4. ConvT bias is `None` (bias=False was respected).\n"
        "5. Input `(N, in_c, H, W)` produces output `(N, out_c, 2*H, 2*W)`.\n"
        "6. `forward` output is non-negative (ReLU).\n"
        "7. `block(x)` (i.e. `__call__`) and `block.forward(x)` agree numerically."
    ),
    "stub_body": (
        "def cx1_make_g_block_cls():\n"
        "    \"\"\"Return the GBlock class (an nn.Module subclass).\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "GBlock = cx1_make_g_block_cls()\n"
        "\n"
        "# Case A: returned a class, not an instance.\n"
        "assert isinstance(GBlock, type), f'must return a class; got {type(GBlock).__name__}'\n"
        "\n"
        "# Case B: an instance is an nn.Module.\n"
        "block = GBlock(in_channels=8, out_channels=4)\n"
        "assert isinstance(block, nn.Module), 'GBlock must subclass nn.Module'\n"
        "\n"
        "# Case C: child modules registered by canonical names + types.\n"
        "children = dict(block.named_children())\n"
        "assert set(children.keys()) == {'convt', 'bn', 'act'}, (\n"
        "    f\"expected named children {{convt,bn,act}}; got {sorted(children.keys())}\"\n"
        ")\n"
        "assert isinstance(children['convt'], nn.ConvTranspose2d), (\n"
        "    f'convt must be nn.ConvTranspose2d; got {type(children[\"convt\"]).__name__}'\n"
        ")\n"
        "assert isinstance(children['bn'], nn.BatchNorm2d)\n"
        "assert isinstance(children['act'], nn.ReLU)\n"
        "\n"
        "# Case D: ConvT hyperparams + bias=False.\n"
        "ct = children['convt']\n"
        "assert ct.in_channels == 8 and ct.out_channels == 4\n"
        "assert ct.kernel_size == (4, 4) and ct.stride == (2, 2) and ct.padding == (1, 1)\n"
        "assert ct.bias is None, 'ConvT bias must be None (bias=False)'\n"
        "\n"
        "# Case E: spatial doubling + channel halving on a forward pass.\n"
        "x = t.randn(2, 8, 4, 4)\n"
        "y = block(x)\n"
        "assert y.shape == (2, 4, 8, 8), f'expected (2,4,8,8); got {tuple(y.shape)}'\n"
        "\n"
        "# Case F: ReLU non-negativity.\n"
        "assert (y >= 0).all(), 'output must be non-negative (ReLU)'\n"
        "\n"
        "# Case G: __call__ and .forward agree (subclass uses Module machinery).\n"
        "block.eval()\n"
        "x2 = t.randn(1, 8, 4, 4)\n"
        "with t.no_grad():\n"
        "    a = block(x2)\n"
        "    b = block.forward(x2)\n"
        "assert t.allclose(a, b, atol=1e-7), '__call__ and .forward disagree — likely overrode __call__ instead of forward'\n"
        "\n"
        "# Case H: cross-check vs an equivalent nn.Sequential composition (in eval mode so BN matches).\n"
        "ref = nn.Sequential(\n"
        "    nn.ConvTranspose2d(8, 4, 4, 2, 1, bias=False),\n"
        "    nn.BatchNorm2d(4),\n"
        "    nn.ReLU(inplace=True),\n"
        ")\n"
        "ref[0].load_state_dict(children['convt'].state_dict())\n"
        "ref[1].load_state_dict(children['bn'].state_dict())\n"
        "ref.eval()\n"
        "with t.no_grad():\n"
        "    expected = ref(x2)\n"
        "assert t.allclose(a, expected, atol=1e-6), 'block output disagrees with the canonical ConvT->BN->ReLU sequence'"
    ),
    "solution_body": (
        "def cx1_make_g_block_cls():\n"
        "    class GBlock(nn.Module):\n"
        "        def __init__(self, in_channels, out_channels):\n"
        "            # Atom B (nn-module-subclass): super().__init__() FIRST.\n"
        "            super().__init__()\n"
        "            # Atom A (convtranspose-bn-activation-block): ConvT(bias=False) -> BN -> ReLU.\n"
        "            self.convt = nn.ConvTranspose2d(\n"
        "                in_channels, out_channels,\n"
        "                kernel_size=4, stride=2, padding=1, bias=False,\n"
        "            )\n"
        "            self.bn = nn.BatchNorm2d(out_channels)\n"
        "            self.act = nn.ReLU(inplace=True)\n"
        "\n"
        "        def forward(self, x):\n"
        "            return self.act(self.bn(self.convt(x)))\n"
        "\n"
        "    return GBlock"
    ),
    "solution_notes": (
        "`super().__init__()` wires up `_parameters`, `_modules`, `_buffers` — without it, "
        "attribute assignment still works but the Module machinery (`.parameters()`, "
        "`.state_dict()`, `.to(device)`) is broken. The `bias=False` choice isn't free — the "
        "ConvT bias would just be subtracted away by BN's running mean, so it's wasted "
        "parameters. Keep `inplace=True` on ReLU only when you don't need the pre-activation "
        "value for autograd elsewhere (here it's safe because we don't branch off the conv "
        "output)."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["convtranspose-bn-activation-block", "nn-module-subclass"],
    "lo": (
        "Compose the DCGAN ConvT->BN->ReLU block with the nn.Module subclass discipline "
        "(super().__init__() + named child registration + forward) so the block exposes "
        "named children and behaves identically to the canonical Sequential."
    ),
}


# ===========================================================================
# cx2 — einops.Rearrange in nn.Sequential before ConvT/BN/ReLU.
# ===========================================================================
spec_2 = {
    "atom_ids": ["convtranspose-bn-activation-block", "rearrange-as-sequential-layer"],
    "subtopics": _subs(["convtranspose-bn-activation-block", "rearrange-as-sequential-layer"]),
    "primary_atom": "convtranspose-bn-activation-block",
    "part": "part5",
    "exercise_index": 2,
    "exercise_title": "Rearrange unflattens the latent into a feature map before a ConvT/BN/ReLU block",
    "slug": "rearrange-then-convt-bn-relu",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "A DCGAN generator can start from a flat latent code `z` of shape `(B, C*H*W)` and "
        "lift it directly into a feature map ready for the first ConvT block. Two atoms wire "
        "this together inside a single `nn.Sequential`:\n\n"
        "1. **rearrange-as-sequential-layer** — `einops.layers.torch.Rearrange` is the "
        "*module* form of `einops.rearrange`. It can sit inside `nn.Sequential` as a real "
        "layer, no `forward` lambda required. Common pattern: "
        "`Rearrange('b (c h w) -> b c h w', c=C, h=H, w=W)`.\n"
        "2. **convtranspose-bn-activation-block** — `ConvTranspose2d(kernel=4, stride=2, "
        "padding=1, bias=False) -> BatchNorm2d -> ReLU`, the canonical generator upsampling "
        "unit (spatial *= 2, channels halve).\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "nn.Sequential(\n"
        "    Rearrange('b (c h w) -> b c h w', c=128, h=4, w=4),       # rearrange-as-sequential-layer.\n"
        "    nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),         # convtranspose-bn-activation-block.\n"
        "    nn.BatchNorm2d(64),\n"
        "    nn.ReLU(inplace=True),\n"
        ")\n"
        "```\n\n"
        "Input shape `(B, 128*4*4) = (B, 2048)`; output shape `(B, 64, 8, 8)`. The "
        "`Rearrange` keeps the whole network expressible as one `nn.Sequential` — no custom "
        "`forward` needed to do the `.view(B, C, H, W)`."
    ),
    "prompt_body": (
        "Implement `cx2_build_rearrange_then_g_block(latent_channels=128, h=4, w=4, "
        "out_channels=64)`. Return an `nn.Sequential` with exactly FOUR children, IN ORDER:\n\n"
        "1. `Rearrange('b (c h w) -> b c h w', c=latent_channels, h=h, w=w)` "
        "(atom: rearrange-as-sequential-layer).\n"
        "2. `nn.ConvTranspose2d(latent_channels, out_channels, kernel_size=4, stride=2, "
        "padding=1, bias=False)` (atom: convtranspose-bn-activation-block, layer 1/3).\n"
        "3. `nn.BatchNorm2d(out_channels)` (atom: convtranspose-bn-activation-block, layer 2/3).\n"
        "4. `nn.ReLU(inplace=True)` (atom: convtranspose-bn-activation-block, layer 3/3).\n\n"
        "The test checks:\n"
        "- Returned value is `nn.Sequential` with exactly 4 children.\n"
        "- Child 0 is an `einops.layers.torch.Rearrange` (NOT a lambda module — must be the "
        "real einops layer).\n"
        "- Children 1/2/3 are `ConvTranspose2d`, `BatchNorm2d`, `ReLU`.\n"
        "- `ConvT.bias is None` (bias=False).\n"
        "- Forward pass: input `(B, latent_channels*h*w)` produces `(B, out_channels, 2*h, 2*w)`.\n"
        "- Output is non-negative.\n"
        "- Numerically agrees with the manual two-step `(rearrange + conv block)` reference."
    ),
    "stub_body": (
        "def cx2_build_rearrange_then_g_block(latent_channels=128, h=4, w=4, out_channels=64):\n"
        "    \"\"\"Return nn.Sequential(Rearrange, ConvT2d, BN2d, ReLU).\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "from einops.layers.torch import Rearrange\n"
        "\n"
        "# Case A: structure.\n"
        "seq = cx2_build_rearrange_then_g_block(latent_channels=128, h=4, w=4, out_channels=64)\n"
        "assert isinstance(seq, nn.Sequential), f'must return nn.Sequential; got {type(seq).__name__}'\n"
        "kids = list(seq.children())\n"
        "assert len(kids) == 4, f'expected 4 children; got {len(kids)}'\n"
        "assert isinstance(kids[0], Rearrange), f'child 0 must be einops Rearrange; got {type(kids[0]).__name__}'\n"
        "assert isinstance(kids[1], nn.ConvTranspose2d)\n"
        "assert isinstance(kids[2], nn.BatchNorm2d)\n"
        "assert isinstance(kids[3], nn.ReLU)\n"
        "\n"
        "# Case B: ConvT hyperparams.\n"
        "ct = kids[1]\n"
        "assert ct.in_channels == 128 and ct.out_channels == 64\n"
        "assert ct.kernel_size == (4, 4) and ct.stride == (2, 2) and ct.padding == (1, 1)\n"
        "assert ct.bias is None, 'ConvT bias must be None (bias=False)'\n"
        "\n"
        "# Case C: forward shape — input is FLAT, output is a 2x-upsampled feature map.\n"
        "seq.eval()\n"
        "B = 3\n"
        "z = t.randn(B, 128 * 4 * 4)\n"
        "with t.no_grad():\n"
        "    y = seq(z)\n"
        "assert y.shape == (B, 64, 8, 8), f'expected (3,64,8,8); got {tuple(y.shape)}'\n"
        "\n"
        "# Case D: ReLU non-negativity.\n"
        "assert (y >= 0).all()\n"
        "\n"
        "# Case E: numerical agreement with the manual two-step composition.\n"
        "manual = nn.Sequential(\n"
        "    nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),\n"
        "    nn.BatchNorm2d(64),\n"
        "    nn.ReLU(inplace=True),\n"
        ")\n"
        "manual[0].load_state_dict(kids[1].state_dict())\n"
        "manual[1].load_state_dict(kids[2].state_dict())\n"
        "manual.eval()\n"
        "z2 = t.randn(B, 128 * 4 * 4)\n"
        "with t.no_grad():\n"
        "    expected = manual(z2.view(B, 128, 4, 4))\n"
        "    got = seq(z2)\n"
        "assert t.allclose(got, expected, atol=1e-6), 'sequential disagrees with manual rearrange+block'\n"
        "\n"
        "# Case F: works for a different (latent_channels, h, w) to confirm the Rearrange picks up the kwargs.\n"
        "seq2 = cx2_build_rearrange_then_g_block(latent_channels=32, h=2, w=2, out_channels=16)\n"
        "z3 = t.randn(2, 32 * 2 * 2)\n"
        "seq2.eval()\n"
        "with t.no_grad():\n"
        "    y2 = seq2(z3)\n"
        "assert y2.shape == (2, 16, 4, 4), f'small-config shape wrong; got {tuple(y2.shape)}'"
    ),
    "solution_body": (
        "def cx2_build_rearrange_then_g_block(latent_channels=128, h=4, w=4, out_channels=64):\n"
        "    return nn.Sequential(\n"
        "        # Atom A (rearrange-as-sequential-layer): einops layer-module form.\n"
        "        Rearrange('b (c h w) -> b c h w', c=latent_channels, h=h, w=w),\n"
        "        # Atom B (convtranspose-bn-activation-block).\n"
        "        nn.ConvTranspose2d(latent_channels, out_channels,\n"
        "                           kernel_size=4, stride=2, padding=1, bias=False),\n"
        "        nn.BatchNorm2d(out_channels),\n"
        "        nn.ReLU(inplace=True),\n"
        "    )"
    ),
    "solution_notes": (
        "The `Rearrange` MODULE (`einops.layers.torch.Rearrange`) is the key to staying inside "
        "`nn.Sequential`. The plain `einops.rearrange` FUNCTION can't be inserted into a "
        "Sequential (it's not a Module). Common bug: shipping `c=latent_channels` as a default "
        "in the pattern string instead of as a kwarg — the kwarg form is what lets einops verify "
        "shape at runtime and produce a clear error if the latent length is wrong."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["convtranspose-bn-activation-block", "rearrange-as-sequential-layer"],
    "lo": (
        "Compose einops.layers.torch.Rearrange (inside nn.Sequential) with a ConvT/BN/ReLU "
        "block so a flat latent code lifts directly into the first upsampled feature map "
        "without a custom forward method."
    ),
}


# ===========================================================================
# cx3 — nn.Module subclass holding an einops Rearrange as a member.
# ===========================================================================
spec_3 = {
    "atom_ids": ["nn-module-subclass", "rearrange-as-sequential-layer"],
    "subtopics": _subs(["nn-module-subclass", "rearrange-as-sequential-layer"]),
    "primary_atom": "nn-module-subclass",
    "part": "part5",
    "exercise_index": 3,
    "exercise_title": "Patchify subclass: nn.Module wrapping an einops Rearrange + Linear head",
    "slug": "patchify-module-with-rearrange-member",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "An `nn.Module` subclass can hold an `einops.layers.torch.Rearrange` as a regular "
        "child module — just assign it in `__init__` and it gets registered (visible in "
        "`.children()`, moves with `.to(device)`). Two atoms compose:\n\n"
        "1. **nn-module-subclass** — define `class Patchify(nn.Module)`, call "
        "`super().__init__()`, register children by attribute assignment, implement "
        "`forward`.\n"
        "2. **rearrange-as-sequential-layer** — instantiate `Rearrange('b c (h p1) (w p2) -> "
        "b (h w) (p1 p2 c)', p1=patch, p2=patch)` and store it on `self`. This is the "
        "Vision-Transformer-style patchify step in module form.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "class Patchify(nn.Module):\n"
        "    def __init__(self, in_channels, patch, embed_dim):\n"
        "        super().__init__()                            # nn-module-subclass.\n"
        "        self.split = Rearrange(                       # rearrange-as-sequential-layer.\n"
        "            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',\n"
        "            p1=patch, p2=patch,\n"
        "        )\n"
        "        self.proj = nn.Linear(patch * patch * in_channels, embed_dim)\n"
        "    def forward(self, x):\n"
        "        return self.proj(self.split(x))\n"
        "```\n\n"
        "**Why store `Rearrange` as `self.split` and not call `einops.rearrange` inside "
        "`forward`.** Three reasons: (a) it's the same parameter-free op in both cases, but "
        "the module form shows up in `print(model)` and `model.named_children()`, which "
        "makes debugging easier; (b) the pattern string + axis kwargs are validated ONCE at "
        "construction time, not on every forward; (c) the module discipline scales — when you "
        "later want to `nn.Sequential(Patchify(...), TransformerBlock(...))`, the subclass "
        "fits in seamlessly."
    ),
    "prompt_body": (
        "Implement `cx3_make_patchify_cls()` — return a `Patchify` class.\n\n"
        "Contract:\n"
        "- `Patchify(in_channels: int, patch: int, embed_dim: int)`.\n"
        "- Inside `__init__`, call `super().__init__()` first (atom: nn-module-subclass).\n"
        "- Register exactly two children by attribute assignment:\n"
        "  - `self.split = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch, p2=patch)` "
        "(atom: rearrange-as-sequential-layer).\n"
        "  - `self.proj = nn.Linear(patch * patch * in_channels, embed_dim)`.\n"
        "- `forward(self, x)` returns `self.proj(self.split(x))`.\n\n"
        "The test checks:\n"
        "1. Returned value is a class.\n"
        "2. Instances are `nn.Module`s.\n"
        "3. `named_children()` is exactly `{'split', 'proj'}`.\n"
        "4. `self.split` is an `einops.layers.torch.Rearrange`.\n"
        "5. `self.proj` is `nn.Linear` with the correct in/out features.\n"
        "6. Forward: `(B, C, H, W)` -> `(B, (H/patch)*(W/patch), embed_dim)` for divisible "
        "shapes.\n"
        "7. The output of `forward(x)` matches the manual reference "
        "`self.proj(Rearrange(...)(x))`.\n"
        "8. `model.parameters()` contains `proj.weight` and `proj.bias` (Rearrange has no "
        "learnable params)."
    ),
    "stub_body": (
        "def cx3_make_patchify_cls():\n"
        "    \"\"\"Return the Patchify class.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "from einops.layers.torch import Rearrange\n"
        "\n"
        "Patchify = cx3_make_patchify_cls()\n"
        "assert isinstance(Patchify, type), 'must return a class'\n"
        "\n"
        "in_c, patch, embed = 3, 4, 16\n"
        "mod = Patchify(in_channels=in_c, patch=patch, embed_dim=embed)\n"
        "assert isinstance(mod, nn.Module), 'Patchify must subclass nn.Module'\n"
        "\n"
        "# Case A: named children — exactly 'split' and 'proj'.\n"
        "kids = dict(mod.named_children())\n"
        "assert set(kids.keys()) == {'split', 'proj'}, (\n"
        "    f\"expected children {{'split','proj'}}; got {sorted(kids.keys())}\"\n"
        ")\n"
        "assert isinstance(kids['split'], Rearrange), (\n"
        "    f'self.split must be einops Rearrange; got {type(kids[\"split\"]).__name__}'\n"
        ")\n"
        "assert isinstance(kids['proj'], nn.Linear)\n"
        "assert kids['proj'].in_features == patch * patch * in_c\n"
        "assert kids['proj'].out_features == embed\n"
        "\n"
        "# Case B: forward shape on a divisible input.\n"
        "B, H, W = 2, 8, 8  # 2x2 grid of 4x4 patches per image.\n"
        "x = t.randn(B, in_c, H, W)\n"
        "mod.eval()\n"
        "with t.no_grad():\n"
        "    y = mod(x)\n"
        "n_patches = (H // patch) * (W // patch)\n"
        "assert y.shape == (B, n_patches, embed), f'expected (2,4,16); got {tuple(y.shape)}'\n"
        "\n"
        "# Case C: matches the manual reference.\n"
        "with t.no_grad():\n"
        "    ref = kids['proj'](kids['split'](x))\n"
        "assert t.allclose(y, ref, atol=1e-6), 'forward output disagrees with manual proj(split(x))'\n"
        "\n"
        "# Case D: Rearrange has NO learnable params; proj has weight + bias.\n"
        "param_names = sorted(name for name, _ in mod.named_parameters())\n"
        "assert param_names == ['proj.bias', 'proj.weight'], (\n"
        "    f'expected only proj.weight, proj.bias; got {param_names}'\n"
        ")\n"
        "\n"
        "# Case E: __call__ vs .forward agreement.\n"
        "x2 = t.randn(1, in_c, H, W)\n"
        "with t.no_grad():\n"
        "    a = mod(x2)\n"
        "    b = mod.forward(x2)\n"
        "assert t.allclose(a, b, atol=1e-7)\n"
        "\n"
        "# Case F: non-divisible input raises (proves Rearrange validates shape, not a manual reshape).\n"
        "try:\n"
        "    _ = mod(t.randn(B, in_c, 7, 7))  # 7 not divisible by patch=4.\n"
        "    raised = False\n"
        "except Exception:\n"
        "    raised = True\n"
        "assert raised, 'non-divisible H/W should raise — Rearrange enforces the patch grid'"
    ),
    "solution_body": (
        "def cx3_make_patchify_cls():\n"
        "    class Patchify(nn.Module):\n"
        "        def __init__(self, in_channels, patch, embed_dim):\n"
        "            # Atom A (nn-module-subclass).\n"
        "            super().__init__()\n"
        "            # Atom B (rearrange-as-sequential-layer): Rearrange as a child Module.\n"
        "            self.split = Rearrange(\n"
        "                'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',\n"
        "                p1=patch, p2=patch,\n"
        "            )\n"
        "            self.proj = nn.Linear(patch * patch * in_channels, embed_dim)\n"
        "\n"
        "        def forward(self, x):\n"
        "            return self.proj(self.split(x))\n"
        "\n"
        "    return Patchify"
    ),
    "solution_notes": (
        "`Rearrange` is registered as a child purely by attribute assignment — the module "
        "machinery in `nn.Module.__setattr__` detects that the RHS is itself a Module. If you "
        "instead stored a `functools.partial(einops.rearrange, ...)`, it would NOT appear in "
        "`children()` and wouldn't move with `.to(device)` (admittedly Rearrange has nothing to "
        "move, but the principle holds for any parameterless layer)."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["nn-module-subclass", "rearrange-as-sequential-layer"],
    "lo": (
        "Compose nn.Module subclassing with einops.layers.torch.Rearrange as a registered "
        "child so a patchify-style reshape participates in named_children, .to(device), and "
        "print(model)."
    ),
}


# ===========================================================================
# cx4 — Latent z -> Linear -> reshape -> ConvT/BN/ReLU upsample stack.
# ===========================================================================
spec_4 = {
    "atom_ids": ["bottleneck-latent-projection", "convtranspose-bn-activation-block"],
    "subtopics": _subs(["bottleneck-latent-projection", "convtranspose-bn-activation-block"]),
    "primary_atom": "bottleneck-latent-projection",
    "part": "part5",
    "exercise_index": 4,
    "exercise_title": "Latent z -> Linear projection -> reshape -> ConvT/BN/ReLU upsample block",
    "slug": "latent-projection-then-convt-block",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "The DCGAN / VAE-decoder entry sequence is: a flat latent code `z` of shape "
        "`(B, latent_dim)` becomes a small 3D feature map `(B, C, H, W)` via a learnable "
        "Linear projection + reshape, then the ConvT upsampling stack takes over.\n\n"
        "1. **bottleneck-latent-projection** — `nn.Linear(latent_dim, C*H*W)` followed by a "
        "reshape to `(B, C, H, W)`. The *only* learnable layer at the bottleneck is the "
        "Linear; no activation, no norm at the bottleneck output itself.\n"
        "2. **convtranspose-bn-activation-block** — the canonical "
        "`ConvTranspose2d(k=4,s=2,p=1,bias=False) -> BatchNorm2d -> ReLU` triple that doubles "
        "spatial size each application.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "class Decoder(nn.Module):\n"
        "    def __init__(self, latent_dim, base_c, base_hw):\n"
        "        super().__init__()\n"
        "        self.base_c, self.base_hw = base_c, base_hw\n"
        "        # bottleneck-latent-projection: Linear then implicit reshape in forward.\n"
        "        self.proj = nn.Linear(latent_dim, base_c * base_hw * base_hw)\n"
        "        # convtranspose-bn-activation-block: 2x upsample, channel halving.\n"
        "        self.block = nn.Sequential(\n"
        "            nn.ConvTranspose2d(base_c, base_c // 2, 4, 2, 1, bias=False),\n"
        "            nn.BatchNorm2d(base_c // 2),\n"
        "            nn.ReLU(inplace=True),\n"
        "        )\n"
        "    def forward(self, z):\n"
        "        h = self.proj(z).view(z.size(0), self.base_c, self.base_hw, self.base_hw)\n"
        "        return self.block(h)\n"
        "```\n\n"
        "**Why no activation on the projection.** Conceptually the bottleneck IS the latent "
        "interface — adding an activation would mean the model can never use the full real "
        "line for any latent dimension. The conv block immediately after has its own "
        "non-linearity (`ReLU`), so the activation 'budget' isn't wasted."
    ),
    "prompt_body": (
        "Implement `cx4_make_decoder_cls()` — return a `Decoder` class.\n\n"
        "Contract:\n"
        "- `Decoder(latent_dim: int, base_channels: int, base_hw: int)`.\n"
        "- `super().__init__()` first.\n"
        "- Store `self.base_c = base_channels` and `self.base_hw = base_hw` (the test reads "
        "these to know the post-projection shape).\n"
        "- `self.proj = nn.Linear(latent_dim, base_channels * base_hw * base_hw)` "
        "(atom: bottleneck-latent-projection). NO activation here.\n"
        "- `self.block = nn.Sequential(ConvTranspose2d(base_channels, base_channels // 2, "
        "kernel_size=4, stride=2, padding=1, bias=False), BatchNorm2d(base_channels // 2), "
        "ReLU(inplace=True))` (atom: convtranspose-bn-activation-block).\n"
        "- `forward(self, z)`:\n"
        "  1. `h = self.proj(z)` — shape `(B, base_c * base_hw**2)`.\n"
        "  2. Reshape `h` to `(B, base_c, base_hw, base_hw)`.\n"
        "  3. Return `self.block(h)` — shape `(B, base_c // 2, 2*base_hw, 2*base_hw)`.\n\n"
        "The test checks:\n"
        "- Returned value is a class; instance is an nn.Module.\n"
        "- Children `proj`, `block` (any order).\n"
        "- `proj` has `in_features=latent_dim`, `out_features=base_channels*base_hw**2`.\n"
        "- `block` is `nn.Sequential` with `ConvT2d/BN/ReLU` in order; ConvT `bias is None`.\n"
        "- Forward shape: `(B, latent_dim)` -> `(B, base_channels//2, 2*base_hw, 2*base_hw)`.\n"
        "- The output is the same as `block(proj(z).view(...))` manually.\n"
        "- After the bottleneck Linear, NO activation is applied before the conv block."
    ),
    "stub_body": (
        "def cx4_make_decoder_cls():\n"
        "    \"\"\"Return the Decoder class.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "Decoder = cx4_make_decoder_cls()\n"
        "assert isinstance(Decoder, type)\n"
        "\n"
        "latent_dim, base_c, base_hw = 100, 64, 4\n"
        "dec = Decoder(latent_dim=latent_dim, base_channels=base_c, base_hw=base_hw)\n"
        "assert isinstance(dec, nn.Module)\n"
        "\n"
        "# Case A: named children.\n"
        "kids = dict(dec.named_children())\n"
        "assert set(kids.keys()) == {'proj', 'block'}, (\n"
        "    f\"expected children {{'proj','block'}}; got {sorted(kids.keys())}\"\n"
        ")\n"
        "assert isinstance(kids['proj'], nn.Linear)\n"
        "assert kids['proj'].in_features == latent_dim\n"
        "assert kids['proj'].out_features == base_c * base_hw * base_hw\n"
        "assert isinstance(kids['block'], nn.Sequential)\n"
        "block_kids = list(kids['block'].children())\n"
        "assert len(block_kids) == 3\n"
        "assert isinstance(block_kids[0], nn.ConvTranspose2d)\n"
        "assert isinstance(block_kids[1], nn.BatchNorm2d)\n"
        "assert isinstance(block_kids[2], nn.ReLU)\n"
        "assert block_kids[0].bias is None, 'ConvT bias must be None'\n"
        "assert block_kids[0].in_channels == base_c and block_kids[0].out_channels == base_c // 2\n"
        "\n"
        "# Case B: forward shape.\n"
        "dec.eval()\n"
        "B = 5\n"
        "z = t.randn(B, latent_dim)\n"
        "with t.no_grad():\n"
        "    out = dec(z)\n"
        "assert out.shape == (B, base_c // 2, 2 * base_hw, 2 * base_hw), (\n"
        "    f'expected ({B},{base_c // 2},{2*base_hw},{2*base_hw}); got {tuple(out.shape)}'\n"
        ")\n"
        "\n"
        "# Case C: matches manual proj+reshape+block path.\n"
        "with t.no_grad():\n"
        "    h_manual = kids['proj'](z).view(B, base_c, base_hw, base_hw)\n"
        "    expected = kids['block'](h_manual)\n"
        "assert t.allclose(out, expected, atol=1e-6), 'forward disagrees with manual proj.view.block'\n"
        "\n"
        "# Case D: NO activation between proj and block — proves the bottleneck is bare Linear.\n"
        "# Reasoning: if a ReLU sat between proj and block, post-proj negatives would be zeroed,\n"
        "# so the post-projection feature map would have NO negative entries. Catch the trap by\n"
        "# constructing z that yields some negative proj outputs and confirming the model uses them.\n"
        "with t.no_grad():\n"
        "    h_raw = kids['proj'](z)\n"
        "    assert (h_raw < 0).any(), 'projection output should have some negative entries for random z'\n"
        "    # If a hidden ReLU lurked, replacing the projection with its ReLU version would NOT change the output:\n"
        "    h_relu_view = F.relu(h_raw).view(B, base_c, base_hw, base_hw)\n"
        "    out_if_relu = kids['block'](h_relu_view)\n"
        "    assert not t.allclose(out, out_if_relu, atol=1e-6), (\n"
        "        'output is identical whether or not we ReLU the projection — '\n"
        "        'either you inserted a hidden ReLU between proj and block, or BN is masking it. '\n"
        "        'The bottleneck must be a BARE Linear projection.'\n"
        "    )"
    ),
    "solution_body": (
        "def cx4_make_decoder_cls():\n"
        "    class Decoder(nn.Module):\n"
        "        def __init__(self, latent_dim, base_channels, base_hw):\n"
        "            super().__init__()\n"
        "            self.base_c = base_channels\n"
        "            self.base_hw = base_hw\n"
        "            # Atom A (bottleneck-latent-projection): bare Linear, no activation.\n"
        "            self.proj = nn.Linear(latent_dim, base_channels * base_hw * base_hw)\n"
        "            # Atom B (convtranspose-bn-activation-block): 2x upsample, channels halve.\n"
        "            self.block = nn.Sequential(\n"
        "                nn.ConvTranspose2d(base_channels, base_channels // 2,\n"
        "                                   kernel_size=4, stride=2, padding=1, bias=False),\n"
        "                nn.BatchNorm2d(base_channels // 2),\n"
        "                nn.ReLU(inplace=True),\n"
        "            )\n"
        "\n"
        "        def forward(self, z):\n"
        "            B = z.size(0)\n"
        "            h = self.proj(z).view(B, self.base_c, self.base_hw, self.base_hw)\n"
        "            return self.block(h)\n"
        "\n"
        "    return Decoder"
    ),
    "solution_notes": (
        "Reshaping with `.view(B, C, H, W)` requires `B*C*H*W == proj.out_features`, which is "
        "enforced here by construction. Using `einops.layers.torch.Rearrange` instead (cx2) "
        "would let you skip the `forward` reshape entirely, at the cost of a tiny extra Module. "
        "The 'no activation at the bottleneck' rule is what lets the latent space have a "
        "principled probabilistic interpretation (especially in VAEs, where `z ~ N(mu, "
        "sigma**2)` must be allowed to take any real value)."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["bottleneck-latent-projection", "convtranspose-bn-activation-block"],
    "lo": (
        "Compose a bare Linear bottleneck projection + reshape with the first ConvT/BN/ReLU "
        "upsampling block so a 1D latent code lifts into a 2x-upsampled feature map without "
        "any hidden activation between the projection and the conv stack."
    ),
}


# ===========================================================================
# cx5 — Linear-to-feature-map projection wrapped as an nn.Module subclass.
# ===========================================================================
spec_5 = {
    "atom_ids": ["bottleneck-latent-projection", "nn-module-subclass"],
    "subtopics": _subs(["bottleneck-latent-projection", "nn-module-subclass"]),
    "primary_atom": "bottleneck-latent-projection",
    "part": "part5",
    "exercise_index": 5,
    "exercise_title": "Latent->feature-map projection as a standalone nn.Module subclass",
    "slug": "latent-projection-as-nn-module-subclass",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "A reusable 'project a latent to a feature map' building block deserves its own "
        "`nn.Module` subclass: it factors out the `Linear + view` pair so it can be slotted "
        "into different generators/decoders without rewriting the reshape arithmetic each "
        "time.\n\n"
        "1. **bottleneck-latent-projection** — `nn.Linear(latent_dim, C*H*W)` + reshape to "
        "`(B, C, H, W)`. The reshape lives in `forward`, the parameters live in the Linear.\n"
        "2. **nn-module-subclass** — wrap the pair as `class LatentToFeatureMap(nn.Module): "
        "...` so it has named children, a `forward`, and works with `.parameters()` / "
        "`.state_dict()`.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "class LatentToFeatureMap(nn.Module):\n"
        "    def __init__(self, latent_dim, channels, hw):\n"
        "        super().__init__()                          # nn-module-subclass.\n"
        "        self.channels, self.hw = channels, hw\n"
        "        self.proj = nn.Linear(latent_dim, channels * hw * hw)   # bottleneck-latent-projection.\n"
        "    def forward(self, z):\n"
        "        return self.proj(z).view(z.size(0), self.channels, self.hw, self.hw)\n"
        "```\n\n"
        "**Why not `nn.Sequential(Linear, Reshape)`.** Because `.view` is not a Module, the "
        "reshape has to live in a `forward` somewhere — and the cleanest place is a tiny "
        "subclass like this one (the einops `Rearrange` route is the other clean option, "
        "exercised in cx2/cx3). Going via a subclass also makes the `channels` and `hw` "
        "configurable, and lets you swap in a `Conv2d(1, channels, 1)` later if you want a "
        "learned reshape without changing callers."
    ),
    "prompt_body": (
        "Implement `cx5_make_latent_to_feature_map_cls()` — return a `LatentToFeatureMap` "
        "class.\n\n"
        "Contract:\n"
        "- `LatentToFeatureMap(latent_dim: int, channels: int, hw: int)`.\n"
        "- `super().__init__()` first (atom: nn-module-subclass).\n"
        "- Store `self.channels = channels` and `self.hw = hw`.\n"
        "- `self.proj = nn.Linear(latent_dim, channels * hw * hw)` "
        "(atom: bottleneck-latent-projection). NO activation, no normalization on this "
        "module — it is JUST a learnable projection + reshape.\n"
        "- `forward(self, z)` returns "
        "`self.proj(z).view(z.size(0), self.channels, self.hw, self.hw)`.\n\n"
        "The test checks:\n"
        "- Class, instance is `nn.Module`.\n"
        "- Exactly one child named `proj`, of type `nn.Linear`, with correct shapes.\n"
        "- `self.channels` and `self.hw` attributes are accessible (the test reads them).\n"
        "- Forward: `(B, latent_dim) -> (B, channels, hw, hw)`.\n"
        "- `forward(z)` numerically equals `mod.proj(z).view(B, channels, hw, hw)`.\n"
        "- `model.parameters()` returns ONLY `proj.weight` and `proj.bias`.\n"
        "- Output preserves the sign of negative entries (i.e. no hidden activation)."
    ),
    "stub_body": (
        "def cx5_make_latent_to_feature_map_cls():\n"
        "    \"\"\"Return the LatentToFeatureMap class.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "LatentToFeatureMap = cx5_make_latent_to_feature_map_cls()\n"
        "assert isinstance(LatentToFeatureMap, type)\n"
        "\n"
        "latent_dim, ch, hw = 32, 16, 4\n"
        "mod = LatentToFeatureMap(latent_dim=latent_dim, channels=ch, hw=hw)\n"
        "assert isinstance(mod, nn.Module)\n"
        "\n"
        "# Case A: configuration attributes accessible.\n"
        "assert mod.channels == ch, f'expected self.channels={ch}; got {mod.channels}'\n"
        "assert mod.hw == hw, f'expected self.hw={hw}; got {mod.hw}'\n"
        "\n"
        "# Case B: named children — exactly one Linear called 'proj'.\n"
        "kids = dict(mod.named_children())\n"
        "assert set(kids.keys()) == {'proj'}, (\n"
        "    f\"expected only 'proj' child; got {sorted(kids.keys())}\"\n"
        ")\n"
        "assert isinstance(kids['proj'], nn.Linear)\n"
        "assert kids['proj'].in_features == latent_dim\n"
        "assert kids['proj'].out_features == ch * hw * hw\n"
        "\n"
        "# Case C: parameters — exactly proj.weight and proj.bias.\n"
        "param_names = sorted(name for name, _ in mod.named_parameters())\n"
        "assert param_names == ['proj.bias', 'proj.weight'], (\n"
        "    f'expected only proj.weight, proj.bias; got {param_names}'\n"
        ")\n"
        "\n"
        "# Case D: forward shape and numerical agreement with manual reference.\n"
        "B = 7\n"
        "z = t.randn(B, latent_dim)\n"
        "mod.eval()\n"
        "with t.no_grad():\n"
        "    y = mod(z)\n"
        "    y_ref = kids['proj'](z).view(B, ch, hw, hw)\n"
        "assert y.shape == (B, ch, hw, hw), f'expected ({B},{ch},{hw},{hw}); got {tuple(y.shape)}'\n"
        "assert t.allclose(y, y_ref, atol=1e-7)\n"
        "\n"
        "# Case E: no hidden activation — negative projection entries pass through with sign intact.\n"
        "with t.no_grad():\n"
        "    raw = kids['proj'](z)\n"
        "    assert (raw < 0).any(), 'random projection should produce some negatives'\n"
        "    # If a ReLU were inserted, y would have NO negatives.\n"
        "    assert (y < 0).any(), 'output has no negatives — likely a hidden ReLU on the projection'\n"
        "\n"
        "# Case F: __call__ vs .forward agreement (Module discipline).\n"
        "with t.no_grad():\n"
        "    a = mod(z)\n"
        "    b = mod.forward(z)\n"
        "assert t.allclose(a, b, atol=1e-7)\n"
        "\n"
        "# Case G: gradient flows through proj (sanity-check Module registration worked).\n"
        "mod.train()\n"
        "z2 = t.randn(B, latent_dim)\n"
        "out2 = mod(z2)\n"
        "out2.sum().backward()\n"
        "assert kids['proj'].weight.grad is not None and kids['proj'].weight.grad.abs().sum() > 0, (\n"
        "    'no grad on proj.weight — Module registration is broken (likely missing super().__init__())'\n"
        ")"
    ),
    "solution_body": (
        "def cx5_make_latent_to_feature_map_cls():\n"
        "    class LatentToFeatureMap(nn.Module):\n"
        "        def __init__(self, latent_dim, channels, hw):\n"
        "            # Atom B (nn-module-subclass): super().__init__() FIRST so _parameters/_modules wire up.\n"
        "            super().__init__()\n"
        "            self.channels = channels\n"
        "            self.hw = hw\n"
        "            # Atom A (bottleneck-latent-projection): bare Linear to flattened feature map.\n"
        "            self.proj = nn.Linear(latent_dim, channels * hw * hw)\n"
        "\n"
        "        def forward(self, z):\n"
        "            B = z.size(0)\n"
        "            return self.proj(z).view(B, self.channels, self.hw, self.hw)\n"
        "\n"
        "    return LatentToFeatureMap"
    ),
    "solution_notes": (
        "The Case G grad-flow check is the strongest evidence that `super().__init__()` ran "
        "before the `self.proj = ...` line — without it, `nn.Module.__setattr__` doesn't add "
        "`proj` to `self._modules`, so `mod.parameters()` would be empty and `.backward()` "
        "would still 'work' but `proj.weight.grad` would still get populated (because the "
        "tensor itself has `requires_grad=True`). The Case E parameter-list check is what "
        "really nails the subclass discipline."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["bottleneck-latent-projection", "nn-module-subclass"],
    "lo": (
        "Compose a bare Linear bottleneck projection + reshape with the nn.Module subclass "
        "discipline (super().__init__() + named child + forward) so the projection is "
        "reusable, named-printable, and serializable as a state_dict."
    ),
}


# ===========================================================================
# cx6 — Symmetric encoder (Conv/BN/LeakyReLU) <-> decoder (ConvT/BN/ReLU).
# ===========================================================================
spec_6 = {
    "atom_ids": ["encoder-decoder-symmetric", "convtranspose-bn-activation-block"],
    "subtopics": _subs(["encoder-decoder-symmetric", "convtranspose-bn-activation-block"]),
    "primary_atom": "encoder-decoder-symmetric",
    "part": "part5",
    "exercise_index": 6,
    "exercise_title": "Symmetric encoder (Conv/BN/LeakyReLU) and decoder (ConvT/BN/ReLU) stacks",
    "slug": "symmetric-encoder-decoder-conv-convt",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "The DCGAN discriminator-as-encoder and generator-as-decoder share a structural "
        "contract: every downsampling step in the encoder is matched by an upsampling step in "
        "the decoder, and the two stacks are channel-mirrored.\n\n"
        "1. **encoder-decoder-symmetric** — for every encoder layer `Conv2d(c, 2c, k=4, "
        "s=2, p=1)` (spatial /= 2, channels *= 2), the decoder has a mirrored "
        "`ConvTranspose2d(2c, c, k=4, s=2, p=1)` (spatial *= 2, channels /= 2). At the "
        "deepest point both stacks meet at the same shape.\n"
        "2. **convtranspose-bn-activation-block** — the canonical *decoder*-side triple: "
        "`ConvT(k=4,s=2,p=1,bias=False) -> BN -> ReLU`. The encoder-side mirror uses "
        "`Conv2d(k=4,s=2,p=1,bias=False) -> BN -> LeakyReLU(0.2)` (DCGAN's published "
        "asymmetry).\n\n"
        "**Anatomy (single encoder/decoder layer at one stage).**\n"
        "```python\n"
        "# Encoder block: spatial /= 2, channels *= 2.\n"
        "enc_block = nn.Sequential(\n"
        "    nn.Conv2d(c_in, c_out, 4, 2, 1, bias=False),\n"
        "    nn.BatchNorm2d(c_out),\n"
        "    nn.LeakyReLU(0.2, inplace=True),\n"
        ")\n"
        "# Decoder block (mirror): spatial *= 2, channels /= 2.\n"
        "dec_block = nn.Sequential(\n"
        "    nn.ConvTranspose2d(c_out, c_in, 4, 2, 1, bias=False),     # convtranspose-bn-activation-block.\n"
        "    nn.BatchNorm2d(c_in),\n"
        "    nn.ReLU(inplace=True),\n"
        ")\n"
        "```\n\n"
        "**Why the activation asymmetry.** DCGAN found empirically that the discriminator "
        "trains more stably with LeakyReLU (no dead-neuron problem on negative inputs) while "
        "the generator wants crisp ReLU outputs. The *block structure* is symmetric — only "
        "the activation differs."
    ),
    "prompt_body": (
        "Implement `cx6_build_symmetric_pair(channel_list)`. `channel_list` is a list like "
        "`[in_c, c1, c2, c3]` describing the encoder's channel progression: the encoder "
        "uses `Conv2d(channel_list[i], channel_list[i+1])` for each adjacent pair.\n\n"
        "Return a tuple `(encoder, decoder)` of two `nn.Sequential`s:\n\n"
        "- `encoder`: for each adjacent pair `(c_in, c_out)` in `channel_list`, append one "
        "block: `Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False) -> "
        "BatchNorm2d(c_out) -> LeakyReLU(0.2, inplace=True)`.\n"
        "- `decoder`: for each adjacent pair `(c_out, c_in)` in REVERSED `channel_list` "
        "(mirroring the encoder), append one block: `ConvTranspose2d(c_out, c_in, "
        "kernel_size=4, stride=2, padding=1, bias=False) -> BatchNorm2d(c_in) -> "
        "ReLU(inplace=True)` "
        "(atom: convtranspose-bn-activation-block, applied at each mirrored stage).\n\n"
        "If `channel_list = [3, 16, 32]`:\n"
        "- encoder has 2 blocks: `(3->16, BN16, LReLU)`, `(16->32, BN32, LReLU)`.\n"
        "- decoder has 2 blocks: `(32->16, BN16, ReLU)`, `(16->3, BN3, ReLU)`.\n\n"
        "Total Sequential children = `3 * (len(channel_list) - 1)` per stack (since each "
        "block contributes 3 layers).\n\n"
        "The test checks:\n"
        "- Both returned objects are `nn.Sequential`.\n"
        "- Encoder Conv layers and Decoder ConvT layers are channel-mirrored "
        "(atom: encoder-decoder-symmetric).\n"
        "- Encoder uses `LeakyReLU(negative_slope=0.2)`, decoder uses `ReLU`.\n"
        "- All Conv/ConvT have `bias is None` (BN follows).\n"
        "- Round-trip shape: `(B, in_c, H, W) -> encoder -> ... -> decoder -> (B, in_c, H, W)`. "
        "Use `H = W = 2**(len(channel_list) - 1)` so each `/2` stage stays integer.\n"
        "- Round-trip works on a multiple-of-2 spatial size."
    ),
    "stub_body": (
        "def cx6_build_symmetric_pair(channel_list):\n"
        "    \"\"\"Return (encoder, decoder) as a pair of nn.Sequentials.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "channel_list = [3, 16, 32]\n"
        "enc, dec = cx6_build_symmetric_pair(channel_list)\n"
        "assert isinstance(enc, nn.Sequential), f'encoder must be nn.Sequential; got {type(enc).__name__}'\n"
        "assert isinstance(dec, nn.Sequential), f'decoder must be nn.Sequential; got {type(dec).__name__}'\n"
        "\n"
        "# Case A: total layer count matches expected (3 layers per block).\n"
        "n_blocks = len(channel_list) - 1\n"
        "enc_layers = list(enc.children())\n"
        "dec_layers = list(dec.children())\n"
        "assert len(enc_layers) == 3 * n_blocks, (\n"
        "    f'expected {3 * n_blocks} encoder layers; got {len(enc_layers)}'\n"
        ")\n"
        "assert len(dec_layers) == 3 * n_blocks, (\n"
        "    f'expected {3 * n_blocks} decoder layers; got {len(dec_layers)}'\n"
        ")\n"
        "\n"
        "# Case B: encoder blocks are Conv2d/BN/LeakyReLU; decoder blocks are ConvT2d/BN/ReLU.\n"
        "for i in range(n_blocks):\n"
        "    e0, e1, e2 = enc_layers[3*i:3*i+3]\n"
        "    assert isinstance(e0, nn.Conv2d), f'encoder layer {3*i} should be Conv2d; got {type(e0).__name__}'\n"
        "    assert isinstance(e1, nn.BatchNorm2d), f'encoder layer {3*i+1} should be BN; got {type(e1).__name__}'\n"
        "    assert isinstance(e2, nn.LeakyReLU), f'encoder layer {3*i+2} should be LeakyReLU; got {type(e2).__name__}'\n"
        "    assert abs(e2.negative_slope - 0.2) < 1e-9, f'LeakyReLU slope must be 0.2; got {e2.negative_slope}'\n"
        "    assert e0.bias is None, f'encoder Conv {i} bias must be None (bias=False)'\n"
        "    assert e0.kernel_size == (4, 4) and e0.stride == (2, 2) and e0.padding == (1, 1)\n"
        "    assert e0.in_channels == channel_list[i] and e0.out_channels == channel_list[i+1], (\n"
        "        f'encoder Conv {i} channels expected {(channel_list[i], channel_list[i+1])}; '\n"
        "        f'got {(e0.in_channels, e0.out_channels)}'\n"
        "    )\n"
        "\n"
        "rev = list(reversed(channel_list))\n"
        "for i in range(n_blocks):\n"
        "    d0, d1, d2 = dec_layers[3*i:3*i+3]\n"
        "    assert isinstance(d0, nn.ConvTranspose2d), f'decoder layer {3*i} should be ConvT2d; got {type(d0).__name__}'\n"
        "    assert isinstance(d1, nn.BatchNorm2d)\n"
        "    assert isinstance(d2, nn.ReLU), f'decoder layer {3*i+2} should be ReLU; got {type(d2).__name__}'\n"
        "    assert d0.bias is None, f'decoder ConvT {i} bias must be None'\n"
        "    assert d0.kernel_size == (4, 4) and d0.stride == (2, 2) and d0.padding == (1, 1)\n"
        "    # Channels are reversed: (c_out, c_in) = (rev[i], rev[i+1]).\n"
        "    assert d0.in_channels == rev[i] and d0.out_channels == rev[i+1], (\n"
        "        f'decoder ConvT {i} channels expected {(rev[i], rev[i+1])}; '\n"
        "        f'got {(d0.in_channels, d0.out_channels)}'\n"
        "    )\n"
        "\n"
        "# Case C: encoder out-channels at stage i == decoder in-channels at stage n_blocks-1-i (atom: encoder-decoder-symmetric).\n"
        "for i in range(n_blocks):\n"
        "    enc_conv = enc_layers[3*i]\n"
        "    dec_convt = dec_layers[3*(n_blocks-1-i)]\n"
        "    assert enc_conv.out_channels == dec_convt.in_channels, (\n"
        "        f'symmetry broken at stage {i}: enc.out_c={enc_conv.out_channels} vs '\n"
        "        f'dec[mirror].in_c={dec_convt.in_channels}'\n"
        "    )\n"
        "    assert enc_conv.in_channels == dec_convt.out_channels, (\n"
        "        f'symmetry broken at stage {i}: enc.in_c={enc_conv.in_channels} vs '\n"
        "        f'dec[mirror].out_c={dec_convt.out_channels}'\n"
        "    )\n"
        "\n"
        "# Case D: round-trip shape — encode then decode brings us back to the input shape.\n"
        "H = W = 2 ** n_blocks  # 4 here, so 4 -> 2 -> 1 after two encoder blocks.\n"
        "B = 2\n"
        "x = t.randn(B, channel_list[0], H, W)\n"
        "enc.eval(); dec.eval()\n"
        "with t.no_grad():\n"
        "    h = enc(x)\n"
        "    expected_deep = (B, channel_list[-1], H // (2 ** n_blocks), W // (2 ** n_blocks))\n"
        "    assert h.shape == expected_deep, (\n"
        "        f'encoder deepest shape expected {expected_deep}; got {tuple(h.shape)}'\n"
        "    )\n"
        "    out = dec(h)\n"
        "assert out.shape == x.shape, (\n"
        "    f'round-trip shape broken; in {tuple(x.shape)} != out {tuple(out.shape)}'\n"
        ")\n"
        "\n"
        "# Case E: longer chain works too — channel_list = [1, 8, 16, 32].\n"
        "enc2, dec2 = cx6_build_symmetric_pair([1, 8, 16, 32])\n"
        "x2 = t.randn(1, 1, 8, 8)\n"
        "enc2.eval(); dec2.eval()\n"
        "with t.no_grad():\n"
        "    out2 = dec2(enc2(x2))\n"
        "assert out2.shape == x2.shape, f'longer chain round-trip broken; got {tuple(out2.shape)}'"
    ),
    "solution_body": (
        "def cx6_build_symmetric_pair(channel_list):\n"
        "    # Atom A (encoder-decoder-symmetric): mirror channel pairs across the two stacks.\n"
        "    enc_layers = []\n"
        "    for c_in, c_out in zip(channel_list[:-1], channel_list[1:]):\n"
        "        # Encoder-side block: Conv -> BN -> LeakyReLU (DCGAN discriminator side).\n"
        "        enc_layers.append(nn.Conv2d(c_in, c_out, 4, 2, 1, bias=False))\n"
        "        enc_layers.append(nn.BatchNorm2d(c_out))\n"
        "        enc_layers.append(nn.LeakyReLU(0.2, inplace=True))\n"
        "    encoder = nn.Sequential(*enc_layers)\n"
        "\n"
        "    # Atom B (convtranspose-bn-activation-block) applied at each mirrored stage.\n"
        "    dec_layers = []\n"
        "    rev = list(reversed(channel_list))\n"
        "    for c_out, c_in in zip(rev[:-1], rev[1:]):\n"
        "        # Decoder-side block: ConvT -> BN -> ReLU (DCGAN generator side).\n"
        "        dec_layers.append(nn.ConvTranspose2d(c_out, c_in, 4, 2, 1, bias=False))\n"
        "        dec_layers.append(nn.BatchNorm2d(c_in))\n"
        "        dec_layers.append(nn.ReLU(inplace=True))\n"
        "    decoder = nn.Sequential(*dec_layers)\n"
        "\n"
        "    return encoder, decoder"
    ),
    "solution_notes": (
        "The encoder/decoder asymmetry is JUST the activation (`LeakyReLU(0.2)` vs `ReLU`) "
        "and the conv direction (`Conv2d` vs `ConvTranspose2d`). Channel widths and "
        "kernel/stride/padding mirror exactly. `bias=False` is correct on both sides because "
        "BN immediately follows. With `k=4, s=2, p=1`, encoder Conv halves spatial size "
        "(`H_out = (H + 2*p - k) / s + 1 = (H - 2) / 2 + 1 = H/2`) and decoder ConvT doubles "
        "it (`H_out = (H - 1) * s - 2*p + k = 2*H`), so the round-trip is exact."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["encoder-decoder-symmetric", "convtranspose-bn-activation-block"],
    "lo": (
        "Compose the symmetric encoder/decoder layout (channel-mirrored stages, opposite "
        "spatial direction) with the canonical ConvT/BN/ReLU decoder block so an arbitrary "
        "channel_list builds matched stacks whose round-trip preserves input shape."
    ),
}


SPECS = [spec_1, spec_2, spec_3, spec_4, spec_5, spec_6]


def main() -> None:
    for spec in SPECS:
        out = emit_composite(spec)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
