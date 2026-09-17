"""Composite drills cx7..cx12 — batch-20 part5 (LL-cell, ARENA VAE/GAN encoder/AE).

Six composite procedural drills exercising 2-atom pairs from ARENA part 5 —
autoencoder/VAE encoder, decoder, bottleneck, ConvT generator blocks, and
how nn.Module + nn.Sequential + Rearrange compose into the AE architecture.

cx7   bottleneck-latent-projection + encoder-decoder-symmetric
        — AE: encoder -> bottleneck -> decoder, end-to-end shape parity
cx8   encoder-decoder-symmetric    + nn-module-subclass
        — symmetric AE packaged as an nn.Module subclass (encoder + decoder children)
cx9   encoder-decoder-symmetric    + rearrange-as-sequential-layer
        — Rearrange layer at the AE bottleneck (B,C,H,W) <-> (B, C*H*W)
cx10  module-composition           + convtranspose-bn-activation-block
        — chain multiple ConvT+BN+ReLU blocks via nn.Sequential
cx11  module-composition           + nn-module-subclass
        — nn.Sequential body held inside an nn.Module subclass
cx12  module-composition           + rearrange-as-sequential-layer
        — Rearrange composed inside a Sequential conv classifier head
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_composite import emit_composite  # noqa: E402

INV = {a["atom_id"]: a for a in json.load(open("/tmp/drill_atoms.json"))}


def _subs(atom_ids):
    return [INV[a]["subtopic"] for a in atom_ids]


# Standard imports — every VAE/GAN/AE composite needs nn / F / Rearrange.
NN_IMPORTS = [
    "import torch.nn as nn",
    "import torch.nn.functional as F",
    "from einops.layers.torch import Rearrange",
]


# ===========================================================================
# cx7 — AE: encoder -> bottleneck -> decoder, shape parity
# ===========================================================================
spec_7 = {
    "atom_ids": ["bottleneck-latent-projection", "encoder-decoder-symmetric"],
    "subtopics": _subs(["bottleneck-latent-projection", "encoder-decoder-symmetric"]),
    "primary_atom": "encoder-decoder-symmetric",
    "part": "part5",
    "exercise_index": 7,
    "exercise_title": "encoder -> bottleneck Linear -> decoder, end-to-end shape parity",
    "slug": "ae-encoder-bottleneck-decoder",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "An MNIST-style autoencoder threads three blocks together:\n"
        "1. **Encoder** — convolutional downsampler that takes `(B, 1, 28, 28)` and produces a "
        "feature map (e.g. `(B, 32, 7, 7)`). Spatial dims shrink by some power of 2.\n"
        "2. **Bottleneck** — flatten the spatial map and project DOWN with a single `nn.Linear` "
        "to `(B, latent_dim)`. This is the `bottleneck-latent-projection` atom: nothing but "
        "a Linear, no activation.\n"
        "3. **Decoder** — mirror image of the encoder: a Linear to project the latent back UP to "
        "the same flattened size, an unflatten/reshape back to `(B, C, H, W)`, then upsampling "
        "convs that mirror each encoder downsample.\n\n"
        "**Why both atoms together.** The bottleneck is the COMPRESSION; the symmetric layout is "
        "the COMMUTATIVE DIAGRAM around it. Without symmetric upsampling you can't reconstruct "
        "back to `(B, 1, 28, 28)`. Without the bottleneck you have no compression at all — just "
        "a fancy conv net.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "# encoder-decoder-symmetric: spatial /= 4 then *= 4.\n"
        "encoder_conv = nn.Sequential(\n"
        "    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14.\n"
        "    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7.\n"
        ")\n"
        "# bottleneck-latent-projection: flatten + Linear, no activation.\n"
        "encode_to_latent = nn.Linear(32 * 7 * 7, latent_dim)\n"
        "decode_from_latent = nn.Linear(latent_dim, 32 * 7 * 7)\n"
        "decoder_conv = nn.Sequential(\n"
        "    nn.Upsample(scale_factor=2),\n"
        "    nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),                    # 7 -> 14.\n"
        "    nn.Upsample(scale_factor=2),\n"
        "    nn.Conv2d(16, 1, 3, padding=1),                                # 14 -> 28.\n"
        ")\n"
        "```\n\n"
        "The test asserts `model(x).shape == x.shape`, that the LATENT shape is `(B, latent_dim)`, "
        "and that the bottleneck Linear has NO nonlinearity after it (negative latents must "
        "survive the encode->decode path)."
    ),
    "prompt_body": (
        "Implement `cx7_make_autoencoder(latent_dim)` — return an instance of a tiny MNIST "
        "autoencoder with three pieces.\n\n"
        "Signature: the returned module accepts `(B, 1, 28, 28)` and returns "
        "`(B, 1, 28, 28)`. It must also expose an `.encode(x)` method that returns "
        "`(B, latent_dim)`.\n\n"
        "Required structure (subclass `nn.Module`):\n"
        "1. `super().__init__()` first.\n"
        "2. `self.encoder_conv = nn.Sequential(`\n"
        "       `nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),`\n"
        "       `nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),`\n"
        "   `)`  # (B, 1, 28, 28) -> (B, 32, 7, 7).\n"
        "3. `self.encode_to_latent = nn.Linear(32 * 7 * 7, latent_dim)`  # bottleneck.\n"
        "4. `self.decode_from_latent = nn.Linear(latent_dim, 32 * 7 * 7)`\n"
        "5. `self.decoder_conv = nn.Sequential(`\n"
        "       `nn.Upsample(scale_factor=2), nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(),`\n"
        "       `nn.Upsample(scale_factor=2), nn.Conv2d(16, 1, kernel_size=3, padding=1),`\n"
        "   `)`  # (B, 32, 7, 7) -> (B, 1, 28, 28).\n"
        "6. `encode(self, x)`:\n"
        "   - run `x` through `encoder_conv` -> `(B, 32, 7, 7)`,\n"
        "   - flatten to `(B, 32*7*7)`,\n"
        "   - apply `encode_to_latent` -> `(B, latent_dim)`, RETURN this (no activation).\n"
        "7. `forward(self, x)`:\n"
        "   - call `self.encode(x)`,\n"
        "   - run latent through `decode_from_latent` -> `(B, 32*7*7)`,\n"
        "   - reshape to `(B, 32, 7, 7)`,\n"
        "   - run through `decoder_conv` -> `(B, 1, 28, 28)` and return.\n\n"
        "No final ReLU/sigmoid on the decoder output — raw linear pixel space.\n\n"
        "The test checks: end-to-end shape parity on multiple batch sizes, latent shape "
        "`(B, latent_dim)`, that `encode_to_latent` has no activation after it (negative "
        "latents pass through), and that all four expected sub-modules exist as named children."
    ),
    "stub_body": (
        "def cx7_make_autoencoder(latent_dim: int):\n"
        "    \"\"\"Return an instance of a tiny conv autoencoder with a Linear bottleneck.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "t.manual_seed(0)\n"
        "model = cx7_make_autoencoder(latent_dim=8)\n"
        "assert isinstance(model, nn.Module)\n"
        "\n"
        "# Case A: required named children all present.\n"
        "named = dict(model.named_children())\n"
        "for key in ['encoder_conv', 'encode_to_latent', 'decode_from_latent', 'decoder_conv']:\n"
        "    assert key in named, f'missing child module {key!r}; got {list(named)}'\n"
        "assert isinstance(named['encode_to_latent'], nn.Linear)\n"
        "assert isinstance(named['decode_from_latent'], nn.Linear)\n"
        "assert named['encode_to_latent'].in_features == 32 * 7 * 7, (\n"
        "    f'bottleneck in_features should be 32*7*7=1568, got {named[\"encode_to_latent\"].in_features}'\n"
        ")\n"
        "assert named['encode_to_latent'].out_features == 8\n"
        "assert named['decode_from_latent'].in_features == 8\n"
        "assert named['decode_from_latent'].out_features == 32 * 7 * 7\n"
        "\n"
        "# Case B: shape parity end-to-end across multiple batch sizes.\n"
        "for B in (1, 3, 8):\n"
        "    x = t.randn(B, 1, 28, 28)\n"
        "    out = model(x)\n"
        "    assert out.shape == x.shape, f'shape parity broken: in={tuple(x.shape)}, out={tuple(out.shape)}'\n"
        "\n"
        "# Case C: encode() returns (B, latent_dim).\n"
        "x = t.randn(5, 1, 28, 28)\n"
        "z = model.encode(x)\n"
        "assert z.shape == (5, 8), f'encode() should return (B, latent_dim)=(5, 8); got {tuple(z.shape)}'\n"
        "\n"
        "# Case D: bottleneck has NO activation — negative latents must pass through.\n"
        "# Construct an input that forces large NEGATIVE pre-activation through the bottleneck.\n"
        "with t.no_grad():\n"
        "    model.encode_to_latent.weight.zero_()\n"
        "    model.encode_to_latent.bias.fill_(-3.7)\n"
        "z2 = model.encode(t.randn(2, 1, 28, 28))\n"
        "# Every entry should be ~-3.7 (no ReLU clipping to 0).\n"
        "assert t.allclose(z2, -3.7 * t.ones_like(z2), atol=1e-5), (\n"
        "    f'bottleneck should have NO activation — negatives must survive. Got max {z2.max().item():.3f}, '\n"
        "    f'min {z2.min().item():.3f}'\n"
        ")\n"
        "\n"
        "# Case E: latent dim is the BOTTLENECK — smaller than 32*7*7.\n"
        "assert 8 < 32 * 7 * 7, 'sanity check: latent dim must be smaller than flattened spatial features'"
    ),
    "solution_body": (
        "def cx7_make_autoencoder(latent_dim: int):\n"
        "    class TinyAE(nn.Module):\n"
        "        def __init__(self, latent_dim):\n"
        "            super().__init__()\n"
        "            # Atom B (encoder-decoder-symmetric): two pool stages -> 28/2/2 = 7.\n"
        "            self.encoder_conv = nn.Sequential(\n"
        "                nn.Conv2d(1, 16, kernel_size=3, padding=1),\n"
        "                nn.ReLU(),\n"
        "                nn.MaxPool2d(2),\n"
        "                nn.Conv2d(16, 32, kernel_size=3, padding=1),\n"
        "                nn.ReLU(),\n"
        "                nn.MaxPool2d(2),\n"
        "            )\n"
        "            # Atom A (bottleneck-latent-projection): bare Linear, no activation.\n"
        "            self.encode_to_latent = nn.Linear(32 * 7 * 7, latent_dim)\n"
        "            self.decode_from_latent = nn.Linear(latent_dim, 32 * 7 * 7)\n"
        "            # Atom B mirrored: two upsamples mirror the two pools.\n"
        "            self.decoder_conv = nn.Sequential(\n"
        "                nn.Upsample(scale_factor=2),\n"
        "                nn.Conv2d(32, 16, kernel_size=3, padding=1),\n"
        "                nn.ReLU(),\n"
        "                nn.Upsample(scale_factor=2),\n"
        "                nn.Conv2d(16, 1, kernel_size=3, padding=1),\n"
        "            )\n"
        "\n"
        "        def encode(self, x):\n"
        "            h = self.encoder_conv(x)            # (B, 32, 7, 7).\n"
        "            h = h.flatten(start_dim=1)          # (B, 32*7*7).\n"
        "            return self.encode_to_latent(h)     # (B, latent_dim).\n"
        "\n"
        "        def forward(self, x):\n"
        "            z = self.encode(x)\n"
        "            h = self.decode_from_latent(z)      # (B, 32*7*7).\n"
        "            h = h.view(h.shape[0], 32, 7, 7)\n"
        "            return self.decoder_conv(h)\n"
        "\n"
        "    return TinyAE(latent_dim)"
    ),
    "solution_notes": (
        "The bottleneck is the *only* place where information has to flow through fewer than "
        "`32*7*7 = 1568` features — that's what makes the AE actually learn a compression. The "
        "`flatten(start_dim=1)` keeps the batch axis intact; `.view(B, 32, 7, 7)` reshapes back "
        "before the upsampling stack. Note: no final ReLU/sigmoid on `decoder_conv` — raw linear "
        "pixel-space output works for normalized images. For [0, 1] pixels you would add a "
        "`nn.Sigmoid()` at the end."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["bottleneck-latent-projection", "encoder-decoder-symmetric"],
    "lo": (
        "Compose a symmetric conv encoder/decoder with a bare-Linear bottleneck so the AE round-trips "
        "(B, 1, 28, 28) -> (B, latent_dim) -> (B, 1, 28, 28) with the latent dim as the compression."
    ),
}


# ===========================================================================
# cx8 — symmetric AE packaged as an nn.Module subclass
# ===========================================================================
spec_8 = {
    "atom_ids": ["encoder-decoder-symmetric", "nn-module-subclass"],
    "subtopics": _subs(["encoder-decoder-symmetric", "nn-module-subclass"]),
    "primary_atom": "encoder-decoder-symmetric",
    "part": "part5",
    "exercise_index": 8,
    "exercise_title": "symmetric AE as an nn.Module subclass (encoder + decoder children)",
    "slug": "symmetric-ae-as-module-subclass",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "An autoencoder is the canonical example of why `nn.Module` subclassing is needed: the "
        "forward pass is *not* just `seq(x)`. It's a SEQUENCE of stages that you may also want to "
        "inspect individually (e.g. visualizing the encoded features). The right pattern is:\n\n"
        "- Hold the **encoder** as one child Module (a `Sequential`, say).\n"
        "- Hold the **decoder** as another child Module.\n"
        "- Subclass `nn.Module`, register both children in `__init__` (after `super().__init__()`), "
        "and write `forward(self, x): return self.decoder(self.encoder(x))`.\n\n"
        "**Why both atoms together.** The symmetric layout is the architecture — it determines the "
        "shape contract `model(x).shape == x.shape`. The module-subclass wrapping is the API — it "
        "gives you `.parameters()`, `.to(device)`, `.train()/.eval()`, and a place to expose "
        "an `.encode(x)` helper. Either alone is incomplete.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "class SymAE(nn.Module):\n"
        "    def __init__(self, c):\n"
        "        super().__init__()                       # nn-module-subclass: required first call.\n"
        "        self.encoder = nn.Sequential(            # encoder-decoder-symmetric: down stages.\n"
        "            nn.Conv2d(c, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),\n"
        "            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),\n"
        "        )\n"
        "        self.decoder = nn.Sequential(            # encoder-decoder-symmetric: up stages.\n"
        "            nn.Upsample(scale_factor=2), nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),\n"
        "            nn.Upsample(scale_factor=2), nn.Conv2d(16, c, 3, padding=1),\n"
        "        )\n"
        "    def forward(self, x):\n"
        "        return self.decoder(self.encoder(x))\n"
        "```"
    ),
    "prompt_body": (
        "Implement `cx8_make_sym_ae_class()` — return the CLASS `SymAE` (not an instance).\n\n"
        "Required structure (`class SymAE(nn.Module)`):\n"
        "1. `__init__(self, in_channels)`:\n"
        "   - `super().__init__()` first.\n"
        "   - `self.encoder = nn.Sequential(`\n"
        "         `nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),`\n"
        "         `nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),`\n"
        "     `)`\n"
        "   - `self.decoder = nn.Sequential(`\n"
        "         `nn.Upsample(scale_factor=2), nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(),`\n"
        "         `nn.Upsample(scale_factor=2), nn.Conv2d(16, in_channels, kernel_size=3, padding=1),`\n"
        "     `)`\n"
        "2. `forward(self, x): return self.decoder(self.encoder(x))`.\n\n"
        "Test checks:\n"
        "- `cx8_make_sym_ae_class()` returns a CLASS (not an instance) that subclasses `nn.Module`.\n"
        "- An instance has BOTH `encoder` and `decoder` registered as named children.\n"
        "- `list(model.parameters())` is non-empty (proves `super().__init__()` was called).\n"
        "- `model(x).shape == x.shape` for several input sizes (spatial dim divisible by 4)."
    ),
    "stub_body": (
        "def cx8_make_sym_ae_class():\n"
        "    \"\"\"Return the SymAE class (not an instance).\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "SymAE = cx8_make_sym_ae_class()\n"
        "assert isinstance(SymAE, type) and issubclass(SymAE, nn.Module), (\n"
        "    'cx8 must return a class that subclasses nn.Module'\n"
        ")\n"
        "\n"
        "# Case A: instantiation + super().__init__() proof.\n"
        "t.manual_seed(0)\n"
        "model = SymAE(in_channels=3)\n"
        "params = list(model.parameters())\n"
        "assert len(params) >= 2, (\n"
        "    f'model has only {len(params)} parameters — did you forget super().__init__()?'\n"
        ")\n"
        "\n"
        "# Case B: encoder + decoder as named children.\n"
        "named = dict(model.named_children())\n"
        "assert 'encoder' in named, f\"expected 'encoder' child; got {list(named)}\"\n"
        "assert 'decoder' in named, f\"expected 'decoder' child; got {list(named)}\"\n"
        "assert isinstance(named['encoder'], nn.Sequential)\n"
        "assert isinstance(named['decoder'], nn.Sequential)\n"
        "\n"
        "# Case C: shape parity across multiple input sizes (must be divisible by 4 spatially).\n"
        "for B, C, H, W in [(1, 3, 8, 8), (2, 3, 12, 16), (4, 3, 32, 32)]:\n"
        "    model_c = SymAE(in_channels=C)\n"
        "    x = t.randn(B, C, H, W)\n"
        "    out = model_c(x)\n"
        "    assert out.shape == x.shape, (\n"
        "        f'shape parity broken: in={tuple(x.shape)}, out={tuple(out.shape)}'\n"
        "    )\n"
        "\n"
        "# Case D: TWO pool stages (encoder), TWO upsample stages (decoder) — symmetric count.\n"
        "encoder_layers = list(model.encoder.children())\n"
        "decoder_layers = list(model.decoder.children())\n"
        "num_pools = sum(1 for layer in encoder_layers if isinstance(layer, nn.MaxPool2d))\n"
        "num_ups = sum(1 for layer in decoder_layers if isinstance(layer, nn.Upsample))\n"
        "assert num_pools == num_ups == 2, (\n"
        "    f'symmetric: #pools ({num_pools}) must equal #upsamples ({num_ups}) == 2'\n"
        ")\n"
        "\n"
        "# Case E: the model is callable via __call__, not just .forward (hooks would break otherwise).\n"
        "x_small = t.randn(1, 3, 8, 8)\n"
        "out_call = model(x_small)\n"
        "out_forward = model.forward(x_small)\n"
        "assert t.allclose(out_call, out_forward, atol=1e-6), (\n"
        "    'module(x) and module.forward(x) should give same numerical result'\n"
        ")"
    ),
    "solution_body": (
        "def cx8_make_sym_ae_class():\n"
        "    class SymAE(nn.Module):\n"
        "        def __init__(self, in_channels):\n"
        "            # Atom B (nn-module-subclass): super() FIRST — wires registry.\n"
        "            super().__init__()\n"
        "            # Atom A (encoder-decoder-symmetric): mirrored down/up stages.\n"
        "            self.encoder = nn.Sequential(\n"
        "                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),\n"
        "                nn.ReLU(),\n"
        "                nn.MaxPool2d(2),\n"
        "                nn.Conv2d(16, 32, kernel_size=3, padding=1),\n"
        "                nn.ReLU(),\n"
        "                nn.MaxPool2d(2),\n"
        "            )\n"
        "            self.decoder = nn.Sequential(\n"
        "                nn.Upsample(scale_factor=2),\n"
        "                nn.Conv2d(32, 16, kernel_size=3, padding=1),\n"
        "                nn.ReLU(),\n"
        "                nn.Upsample(scale_factor=2),\n"
        "                nn.Conv2d(16, in_channels, kernel_size=3, padding=1),\n"
        "            )\n"
        "\n"
        "        def forward(self, x):\n"
        "            return self.decoder(self.encoder(x))\n"
        "\n"
        "    return SymAE"
    ),
    "solution_notes": (
        "Returning the CLASS (not an instance) is the idiom we use when the test needs to "
        "construct multiple instances with different channel counts. The encoder+decoder name "
        "registration lets you do `model.encoder(x)` to inspect the encoded feature map, which is "
        "how reconstruction-quality visualizations are built."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["encoder-decoder-symmetric", "nn-module-subclass"],
    "lo": (
        "Compose nn.Module subclassing (super().__init__ + named encoder/decoder children) with "
        "encoder-decoder symmetric layout so the AE is a single Module that satisfies "
        "model(x).shape == x.shape."
    ),
}


# ===========================================================================
# cx9 — Rearrange layer at the AE bottleneck
# ===========================================================================
spec_9 = {
    "atom_ids": ["encoder-decoder-symmetric", "rearrange-as-sequential-layer"],
    "subtopics": _subs(["encoder-decoder-symmetric", "rearrange-as-sequential-layer"]),
    "primary_atom": "rearrange-as-sequential-layer",
    "part": "part5",
    "exercise_index": 9,
    "exercise_title": "Rearrange layer flattens & restores (B, C, H, W) <-> (B, C*H*W) at the AE bottleneck",
    "slug": "rearrange-bottleneck-flatten-restore",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "An AE bottleneck has to go from `(B, C, H, W)` (the encoder's last feature map) to "
        "`(B, latent_dim)` (a flat code), and back. The boilerplate way is two `view()` / `flatten()` "
        "calls in `forward`. The CLEANER way — and what ARENA uses — is to put "
        "`einops.layers.torch.Rearrange` patterns inside the `Sequential` stacks themselves:\n\n"
        "- Encoder ends with `Rearrange('b c h w -> b (c h w)')` then `Linear(C*H*W, latent_dim)`.\n"
        "- Decoder starts with `Linear(latent_dim, C*H*W)` then `Rearrange('b (c h w) -> b c h w', c=C, h=H, w=W)`.\n\n"
        "Both directions live inside the `Sequential` body — `forward` is just "
        "`self.decoder(self.encoder(x))`. No explicit reshape calls.\n\n"
        "**Why both atoms together.** Without the Rearrange-as-layer trick you'd need a custom "
        "`forward()` that interleaves the conv stack with shape gymnastics. With it, the "
        "encoder/decoder become PURE `nn.Sequential` modules — and the symmetric structure is "
        "visible in the source code (every Rearrange in the encoder is mirrored by an inverse "
        "Rearrange in the decoder).\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "encoder = nn.Sequential(\n"
        "    nn.Conv2d(1, 32, 3, padding=1, stride=2),    # 8 -> 4.\n"
        "    nn.ReLU(),\n"
        "    Rearrange('b c h w -> b (c h w)'),           # rearrange-as-sequential-layer.\n"
        "    nn.Linear(32 * 4 * 4, latent_dim),\n"
        ")\n"
        "decoder = nn.Sequential(\n"
        "    nn.Linear(latent_dim, 32 * 4 * 4),\n"
        "    Rearrange('b (c h w) -> b c h w', c=32, h=4, w=4),\n"
        "    nn.Upsample(scale_factor=2),                 # 4 -> 8.\n"
        "    nn.Conv2d(32, 1, 3, padding=1),\n"
        ")\n"
        "```"
    ),
    "prompt_body": (
        "Implement `cx9_make_rearrange_ae(latent_dim)` — return an `nn.Sequential` that round-trips "
        "`(B, 1, 8, 8)` -> `(B, 1, 8, 8)`, with a Linear bottleneck of size `latent_dim` in the "
        "middle. The WHOLE model must be a single `nn.Sequential` (no custom Module subclass).\n\n"
        "Required layer order (call `nn.Sequential(*layers)`):\n"
        "1. `nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=2)`  # (B, 1, 8, 8) -> (B, 32, 4, 4).\n"
        "2. `nn.ReLU()`\n"
        "3. `Rearrange('b c h w -> b (c h w)')`  # (B, 32, 4, 4) -> (B, 32*4*4).\n"
        "4. `nn.Linear(32 * 4 * 4, latent_dim)`  # bottleneck.\n"
        "5. `nn.Linear(latent_dim, 32 * 4 * 4)`  # un-bottleneck.\n"
        "6. `Rearrange('b (c h w) -> b c h w', c=32, h=4, w=4)`  # (B, 32*4*4) -> (B, 32, 4, 4).\n"
        "7. `nn.Upsample(scale_factor=2)`  # (B, 32, 4, 4) -> (B, 32, 8, 8).\n"
        "8. `nn.Conv2d(32, 1, kernel_size=3, padding=1)`  # (B, 32, 8, 8) -> (B, 1, 8, 8).\n\n"
        "Return the `nn.Sequential` instance.\n\n"
        "Test checks:\n"
        "- Return value is `nn.Sequential` (not a custom Module).\n"
        "- The 3rd layer is a `Rearrange` Module (capital R — the layer form).\n"
        "- The 6th layer is also a `Rearrange` (the inverse).\n"
        "- Round-trip shape: `model(x).shape == (B, 1, 8, 8)` for any batch size.\n"
        "- Latent shape inside the model: peek at intermediate by running the first 4 layers; "
        "result must be `(B, latent_dim)`."
    ),
    "stub_body": (
        "def cx9_make_rearrange_ae(latent_dim: int) -> 'nn.Sequential':\n"
        "    \"\"\"Return an nn.Sequential AE that uses Rearrange layers at the bottleneck.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "model = cx9_make_rearrange_ae(latent_dim=6)\n"
        "assert isinstance(model, nn.Sequential), f'expected nn.Sequential, got {type(model).__name__}'\n"
        "\n"
        "layers = list(model.children())\n"
        "assert len(layers) == 8, f'expected 8 layers; got {len(layers)}: {[type(l).__name__ for l in layers]}'\n"
        "\n"
        "# Case A: layers 3 and 6 are Rearrange.\n"
        "assert isinstance(layers[2], Rearrange), f'layer 3 must be Rearrange, got {type(layers[2]).__name__}'\n"
        "assert isinstance(layers[5], Rearrange), f'layer 6 must be Rearrange, got {type(layers[5]).__name__}'\n"
        "\n"
        "# Case B: end-to-end shape parity (B, 1, 8, 8) -> (B, 1, 8, 8).\n"
        "for B in (1, 4, 7):\n"
        "    x = t.randn(B, 1, 8, 8)\n"
        "    out = model(x)\n"
        "    assert out.shape == x.shape, f'shape parity broken at B={B}: {tuple(out.shape)}'\n"
        "\n"
        "# Case C: intermediate latent shape is (B, latent_dim).\n"
        "encoder_part = nn.Sequential(*layers[:4])\n"
        "x = t.randn(3, 1, 8, 8)\n"
        "z = encoder_part(x)\n"
        "assert z.shape == (3, 6), f'latent shape after first 4 layers should be (3, 6); got {tuple(z.shape)}'\n"
        "\n"
        "# Case D: the second Rearrange is the INVERSE — its output is (B, 32, 4, 4).\n"
        "first_six = nn.Sequential(*layers[:6])\n"
        "feat = first_six(x)\n"
        "assert feat.shape == (3, 32, 4, 4), (\n"
        "    f'after the inverse Rearrange, shape should be (3, 32, 4, 4); got {tuple(feat.shape)}'\n"
        ")\n"
        "\n"
        "# Case E: the Rearrange flatten preserves data (round-trip is lossless when nothing else acts).\n"
        "# Build a tiny throwaway Sequential of just the two Rearrange layers + identity linears.\n"
        "test_rt = nn.Sequential(\n"
        "    Rearrange('b c h w -> b (c h w)'),\n"
        "    Rearrange('b (c h w) -> b c h w', c=32, h=4, w=4),\n"
        ")\n"
        "feat_in = t.randn(2, 32, 4, 4)\n"
        "feat_out = test_rt(feat_in)\n"
        "assert t.allclose(feat_in, feat_out, atol=1e-7), 'Rearrange round-trip should be lossless'"
    ),
    "solution_body": (
        "def cx9_make_rearrange_ae(latent_dim: int):\n"
        "    # Atom B (rearrange-as-sequential-layer): Rearrange goes INSIDE Sequential so\n"
        "    # we never have to write a custom forward(). Atom A (encoder-decoder-symmetric):\n"
        "    # one conv-stride-2 down, one upsample up; one Rearrange flat, one inverse.\n"
        "    return nn.Sequential(\n"
        "        nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=2),\n"
        "        nn.ReLU(),\n"
        "        Rearrange('b c h w -> b (c h w)'),\n"
        "        nn.Linear(32 * 4 * 4, latent_dim),\n"
        "        nn.Linear(latent_dim, 32 * 4 * 4),\n"
        "        Rearrange('b (c h w) -> b c h w', c=32, h=4, w=4),\n"
        "        nn.Upsample(scale_factor=2),\n"
        "        nn.Conv2d(32, 1, kernel_size=3, padding=1),\n"
        "    )"
    ),
    "solution_notes": (
        "Two things to notice. First, the SECOND Rearrange needs the explicit `c=32, h=4, w=4` "
        "kwargs — einops can't infer them from the flat shape alone. Second, the model is just an "
        "`nn.Sequential` — no custom `forward()`, no Module subclass. The whole architecture is "
        "declarative."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["encoder-decoder-symmetric", "rearrange-as-sequential-layer"],
    "lo": (
        "Compose the symmetric encoder/decoder layout with Rearrange-as-Sequential-layer to "
        "flatten + restore at the bottleneck without writing any custom forward()."
    ),
}


# ===========================================================================
# cx10 — chain multiple ConvT+BN+ReLU blocks via nn.Sequential
# ===========================================================================
spec_10 = {
    "atom_ids": ["module-composition", "convtranspose-bn-activation-block"],
    "subtopics": _subs(["module-composition", "convtranspose-bn-activation-block"]),
    "primary_atom": "convtranspose-bn-activation-block",
    "part": "part5",
    "exercise_index": 10,
    "exercise_title": "chain three ConvT+BN+ReLU generator blocks via nn.Sequential",
    "slug": "convt-bn-relu-stack-via-sequential",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "A DCGAN generator is a STACK of ConvT+BN+ReLU blocks (the `convtranspose-bn-activation-block` "
        "atom), one per doubling of spatial size. Each block follows the same recipe:\n"
        "`nn.Sequential(nn.ConvTranspose2d(stride=2, kernel=4, padding=1, bias=False), nn.BatchNorm2d, nn.ReLU)`.\n\n"
        "Composing N of these blocks back-to-back is the `module-composition` atom in action: "
        "you wrap them in an OUTER `nn.Sequential` and the spatial size doubles N times.\n\n"
        "**Why both atoms together.** A single block produces ONE doubling; stacking blocks "
        "produces the full 4 -> 8 -> 16 -> 32 upsampling pipeline that maps a 4x4 noise-derived "
        "feature map to a 32x32 image (or 64x64, or 128x128 with another block). The outer "
        "`Sequential` IS the composition; without it you'd be writing a custom forward.\n\n"
        "**Anatomy (three-block stack: 4x4 -> 32x32).**\n"
        "```python\n"
        "def block(c_in, c_out):\n"
        "    return nn.Sequential(\n"
        "        nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False),\n"
        "        nn.BatchNorm2d(c_out),\n"
        "        nn.ReLU(inplace=True),\n"
        "    )\n"
        "\n"
        "generator = nn.Sequential(\n"
        "    block(256, 128),  # 4x4 -> 8x8.\n"
        "    block(128, 64),   # 8x8 -> 16x16.\n"
        "    block(64, 32),    # 16x16 -> 32x32.\n"
        ")\n"
        "```\n\n"
        "Note: `bias=False` on the ConvT because BN immediately follows (BN subtracts the mean, "
        "rendering the conv bias redundant)."
    ),
    "prompt_body": (
        "Implement `cx10_make_generator_stack(channels)` — return an `nn.Sequential` of THREE "
        "ConvT+BN+ReLU blocks.\n\n"
        "Inputs: `channels` is a list of 4 ints, e.g. `[256, 128, 64, 32]`. The k-th block "
        "transforms `channels[k] -> channels[k+1]` and DOUBLES the spatial dim.\n\n"
        "Each block (you may write a helper) is:\n"
        "```\n"
        "nn.Sequential(\n"
        "    nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False),\n"
        "    nn.BatchNorm2d(c_out),\n"
        "    nn.ReLU(inplace=True),\n"
        ")\n"
        "```\n\n"
        "Then `nn.Sequential` the three blocks together and RETURN that outer Sequential. The "
        "outer Sequential contains 3 child modules (each itself a Sequential of 3 layers).\n\n"
        "Test checks:\n"
        "- Outer return is `nn.Sequential` with exactly 3 children.\n"
        "- Each child is itself a Sequential of 3 layers: ConvTranspose2d, BatchNorm2d, ReLU.\n"
        "- ConvT params: `kernel=4, stride=2, padding=1, bias=None` (bias=False).\n"
        "- Channel chain matches `channels`.\n"
        "- Running a `(1, channels[0], 4, 4)` input through gives `(1, channels[3], 32, 32)` — "
        "spatial dim doubled 3 times."
    ),
    "stub_body": (
        "def cx10_make_generator_stack(channels):\n"
        "    \"\"\"Return nn.Sequential of THREE ConvT+BN+ReLU blocks with channel chain = channels.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "channels = [256, 128, 64, 32]\n"
        "gen = cx10_make_generator_stack(channels)\n"
        "assert isinstance(gen, nn.Sequential), f'outer must be nn.Sequential, got {type(gen).__name__}'\n"
        "\n"
        "blocks = list(gen.children())\n"
        "assert len(blocks) == 3, f'expected 3 child blocks; got {len(blocks)}'\n"
        "\n"
        "# Case A: each block is itself a Sequential of (ConvT, BN, ReLU).\n"
        "for i, blk in enumerate(blocks):\n"
        "    assert isinstance(blk, nn.Sequential), f'block {i} must be nn.Sequential, got {type(blk).__name__}'\n"
        "    layers = list(blk.children())\n"
        "    assert len(layers) == 3, f'block {i} should have 3 layers; got {len(layers)}'\n"
        "    assert isinstance(layers[0], nn.ConvTranspose2d), f'block {i} layer 0 not ConvT'\n"
        "    assert isinstance(layers[1], nn.BatchNorm2d), f'block {i} layer 1 not BatchNorm2d'\n"
        "    assert isinstance(layers[2], nn.ReLU), f'block {i} layer 2 not ReLU'\n"
        "    ct = layers[0]\n"
        "    assert ct.in_channels == channels[i] and ct.out_channels == channels[i + 1], (\n"
        "        f'block {i} channel chain wrong: in={ct.in_channels} (want {channels[i]}), '\n"
        "        f'out={ct.out_channels} (want {channels[i + 1]})'\n"
        "    )\n"
        "    assert ct.kernel_size == (4, 4), f'block {i} kernel should be 4, got {ct.kernel_size}'\n"
        "    assert ct.stride == (2, 2), f'block {i} stride should be 2, got {ct.stride}'\n"
        "    assert ct.padding == (1, 1), f'block {i} padding should be 1, got {ct.padding}'\n"
        "    assert ct.bias is None, f'block {i} ConvT must have bias=False'\n"
        "    bn = layers[1]\n"
        "    assert bn.num_features == channels[i + 1], (\n"
        "        f'block {i} BatchNorm num_features should match ConvT out, got {bn.num_features}'\n"
        "    )\n"
        "\n"
        "# Case B: spatial-doubling end to end.\n"
        "gen.eval()  # BN in eval avoids needing big batch.\n"
        "x = t.randn(2, channels[0], 4, 4)\n"
        "out = gen(x)\n"
        "assert out.shape == (2, channels[-1], 32, 32), (\n"
        "    f'expected output shape (2, {channels[-1]}, 32, 32) after 3 doublings; got {tuple(out.shape)}'\n"
        ")\n"
        "\n"
        "# Case C: total parameter count proves all 3 blocks were registered (not just kept locally).\n"
        "params = list(gen.parameters())\n"
        "# Each block contributes: ConvT weight + BN weight + BN bias = 3 params (ConvT bias=False).\n"
        "# 3 blocks * 3 = 9 params.\n"
        "assert len(params) == 9, (\n"
        "    f'expected 9 params (3 blocks * 3 each: ConvT.weight, BN.weight, BN.bias); got {len(params)}'\n"
        ")"
    ),
    "solution_body": (
        "def cx10_make_generator_stack(channels):\n"
        "    # Atom A (convtranspose-bn-activation-block): the reusable inner block.\n"
        "    def block(c_in, c_out):\n"
        "        return nn.Sequential(\n"
        "            nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False),\n"
        "            nn.BatchNorm2d(c_out),\n"
        "            nn.ReLU(inplace=True),\n"
        "        )\n"
        "\n"
        "    # Atom B (module-composition): wrap the three blocks in an outer Sequential\n"
        "    # so they register as named children (block.0, block.1, block.2).\n"
        "    return nn.Sequential(\n"
        "        block(channels[0], channels[1]),\n"
        "        block(channels[1], channels[2]),\n"
        "        block(channels[2], channels[3]),\n"
        "    )"
    ),
    "solution_notes": (
        "DCGAN's classic generator uses 4 such blocks (4x4 -> 64x64). The factory pattern "
        "(`block(c_in, c_out)`) lets you generate the stack from a channel list of any length. "
        "Sequential-of-Sequentials is fully supported by PyTorch — the outer's `.parameters()` "
        "recursively collects from all nested blocks."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["module-composition", "convtranspose-bn-activation-block"],
    "lo": (
        "Compose multiple ConvT+BN+ReLU blocks into an outer nn.Sequential so spatial doubling "
        "chains across stages and parameters of every nested block register through the outer."
    ),
}


# ===========================================================================
# cx11 — nn.Sequential body held inside an nn.Module subclass
# ===========================================================================
spec_11 = {
    "atom_ids": ["module-composition", "nn-module-subclass"],
    "subtopics": _subs(["module-composition", "nn-module-subclass"]),
    "primary_atom": "module-composition",
    "part": "part5",
    "exercise_index": 11,
    "exercise_title": "nn.Sequential body lives inside an nn.Module subclass",
    "slug": "sequential-inside-module-subclass",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Sometimes you need a custom Module — for a non-Sequential forward (e.g. taking BOTH "
        "`(x, y)` as input, or returning a tuple), or to attach helper methods (`.encode(x)`, "
        "`.generate(n)`). But the bulk of the forward is still a STACK of layers. The idiomatic "
        "pattern is to:\n\n"
        "- Subclass `nn.Module`.\n"
        "- Hold the layer stack as a NAMED ATTRIBUTE that is itself an `nn.Sequential` "
        "(or a list of children wrapped in `nn.Sequential`).\n"
        "- Reference the stack inside `forward` as `self.layers(x)`.\n\n"
        "**Why both atoms together.** The `nn.Module` subclass is the API (you get "
        "`.parameters()`, `.to(device)`, `.train()/.eval()`, and a place for helper methods). "
        "The `nn.Sequential` attribute is the COMPOSITION (you get a clean linear stack without "
        "writing `forward` for each layer).\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "class Generator(nn.Module):\n"
        "    def __init__(self, latent_dim):\n"
        "        super().__init__()                       # nn-module-subclass.\n"
        "        self.project = nn.Linear(latent_dim, 256 * 4 * 4)\n"
        "        self.layers = nn.Sequential(             # module-composition: layer stack as a CHILD.\n"
        "            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),\n"
        "            nn.BatchNorm2d(128),\n"
        "            nn.ReLU(inplace=True),\n"
        "            nn.ConvTranspose2d(128, 1, 4, 2, 1),\n"
        "            nn.Tanh(),\n"
        "        )\n"
        "    def forward(self, z):\n"
        "        h = self.project(z).view(-1, 256, 4, 4)\n"
        "        return self.layers(h)\n"
        "    def generate(self, n):                       # helper only possible because we subclassed.\n"
        "        return self(t.randn(n, self.project.in_features))\n"
        "```"
    ),
    "prompt_body": (
        "Implement `cx11_make_generator_class()` — return a CLASS `Generator(nn.Module)` "
        "implementing a tiny GAN generator with a Linear projection + a Sequential stack.\n\n"
        "Required structure:\n"
        "1. `__init__(self, latent_dim)`:\n"
        "   - `super().__init__()` first.\n"
        "   - `self.project = nn.Linear(latent_dim, 64 * 4 * 4)`  # Linear projection.\n"
        "   - `self.layers = nn.Sequential(`  # the layer stack as a named child.\n"
        "       `nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),`\n"
        "       `nn.BatchNorm2d(32),`\n"
        "       `nn.ReLU(inplace=True),`\n"
        "       `nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),`\n"
        "       `nn.Tanh(),`\n"
        "     `)`  # (B, 64, 4, 4) -> (B, 32, 8, 8) -> (B, 1, 16, 16).\n"
        "2. `forward(self, z)`:\n"
        "   - z is `(B, latent_dim)`.\n"
        "   - `h = self.project(z)` -> `(B, 64*4*4)`.\n"
        "   - reshape to `(B, 64, 4, 4)`.\n"
        "   - return `self.layers(h)` -> `(B, 1, 16, 16)`.\n"
        "3. `generate(self, n)` — convenience method:\n"
        "   - Sample `z = t.randn(n, self.project.in_features)`.\n"
        "   - Return `self(z)` (uses `__call__`, not `.forward`).\n\n"
        "Test checks:\n"
        "- `cx11_make_generator_class()` returns a class subclassing `nn.Module`.\n"
        "- Instance has both `project` (Linear) and `layers` (Sequential) as named children.\n"
        "- `forward(z)` produces `(B, 1, 16, 16)`.\n"
        "- `generate(n)` produces `(n, 1, 16, 16)` and uses `__call__`.\n"
        "- All parameters from BOTH `project` and `layers` show up in `.parameters()`."
    ),
    "stub_body": (
        "def cx11_make_generator_class():\n"
        "    \"\"\"Return the Generator class.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "Generator = cx11_make_generator_class()\n"
        "assert isinstance(Generator, type) and issubclass(Generator, nn.Module)\n"
        "\n"
        "# Case A: instantiation + super().__init__() proof.\n"
        "t.manual_seed(0)\n"
        "gen = Generator(latent_dim=10)\n"
        "params = list(gen.parameters())\n"
        "assert len(params) > 2, (\n"
        "    f'too few params ({len(params)}) — did you forget super().__init__()?'\n"
        ")\n"
        "\n"
        "# Case B: named children — project AND layers.\n"
        "named = dict(gen.named_children())\n"
        "assert 'project' in named and isinstance(named['project'], nn.Linear), (\n"
        "    f\"expected 'project' Linear child; got {list(named)}\"\n"
        ")\n"
        "assert 'layers' in named and isinstance(named['layers'], nn.Sequential), (\n"
        "    f\"expected 'layers' Sequential child; got {list(named)}\"\n"
        ")\n"
        "assert named['project'].in_features == 10\n"
        "assert named['project'].out_features == 64 * 4 * 4\n"
        "\n"
        "# Case C: forward shape.\n"
        "z = t.randn(3, 10)\n"
        "out = gen(z)\n"
        "assert out.shape == (3, 1, 16, 16), f'expected (3, 1, 16, 16); got {tuple(out.shape)}'\n"
        "\n"
        "# Case D: layers stack has the right composition (5 layers).\n"
        "layer_seq = list(named['layers'].children())\n"
        "assert len(layer_seq) == 5, f'layers should have 5 inner modules; got {len(layer_seq)}'\n"
        "assert isinstance(layer_seq[0], nn.ConvTranspose2d)\n"
        "assert isinstance(layer_seq[1], nn.BatchNorm2d)\n"
        "assert isinstance(layer_seq[2], nn.ReLU)\n"
        "assert isinstance(layer_seq[3], nn.ConvTranspose2d)\n"
        "assert isinstance(layer_seq[4], nn.Tanh)\n"
        "\n"
        "# Case E: generate(n) shape and that it uses __call__.\n"
        "t.manual_seed(1)\n"
        "samples = gen.generate(5)\n"
        "assert samples.shape == (5, 1, 16, 16), f'expected (5, 1, 16, 16); got {tuple(samples.shape)}'\n"
        "\n"
        "# Case F: parameters from BOTH children are recursively collected.\n"
        "# project: Linear -> 2 params (weight + bias).\n"
        "# layers: ConvT(bias=F)+BN(2) + ConvT(bias=T)+Tanh(0) = 1+2+2 = 5 params.\n"
        "# total = 7.\n"
        "assert len(params) == 7, f'expected 7 total params (2 project + 5 layers); got {len(params)}'"
    ),
    "solution_body": (
        "def cx11_make_generator_class():\n"
        "    class Generator(nn.Module):\n"
        "        def __init__(self, latent_dim):\n"
        "            # Atom B (nn-module-subclass): wire up the module registry first.\n"
        "            super().__init__()\n"
        "            self.project = nn.Linear(latent_dim, 64 * 4 * 4)\n"
        "            # Atom A (module-composition): nn.Sequential held as a named child.\n"
        "            self.layers = nn.Sequential(\n"
        "                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),\n"
        "                nn.BatchNorm2d(32),\n"
        "                nn.ReLU(inplace=True),\n"
        "                nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),\n"
        "                nn.Tanh(),\n"
        "            )\n"
        "\n"
        "        def forward(self, z):\n"
        "            h = self.project(z)\n"
        "            h = h.view(-1, 64, 4, 4)\n"
        "            return self.layers(h)\n"
        "\n"
        "        def generate(self, n):\n"
        "            z = t.randn(n, self.project.in_features)\n"
        "            return self(z)  # __call__, not .forward — runs hooks.\n"
        "\n"
        "    return Generator"
    ),
    "solution_notes": (
        "Notice `self.layers` is a single attribute holding the WHOLE Sequential — child-of-child "
        "registration works recursively, so `gen.parameters()` finds the ConvT weights inside "
        "`layers` without any extra wiring. The `generate(n)` helper is the payoff: it's not "
        "possible to attach helper methods to a bare `nn.Sequential`, so when you want them, you "
        "subclass."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["module-composition", "nn-module-subclass"],
    "lo": (
        "Compose an nn.Module subclass (super().__init__ + helper methods like generate()) with "
        "module-composition (nn.Sequential held as a named child attribute) to build a generator "
        "whose layers register transitively through the outer module."
    ),
}


# ===========================================================================
# cx12 — Rearrange composed inside a Sequential conv classifier head
# ===========================================================================
spec_12 = {
    "atom_ids": ["module-composition", "rearrange-as-sequential-layer"],
    "subtopics": _subs(["module-composition", "rearrange-as-sequential-layer"]),
    "primary_atom": "rearrange-as-sequential-layer",
    "part": "part5",
    "exercise_index": 12,
    "exercise_title": "Rearrange-as-layer composes inside a nested Sequential classifier head",
    "slug": "rearrange-inside-nested-sequential",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "ARENA's discriminator (and almost every conv classifier) ends with a `(B, C, H, W) -> "
        "(B, num_classes)` head. The head IS a sub-Sequential composed inside the outer model:\n\n"
        "```python\n"
        "head = nn.Sequential(\n"
        "    Rearrange('b c h w -> b (c h w)'),       # rearrange-as-sequential-layer (flatten).\n"
        "    nn.Linear(C * H * W, num_classes),\n"
        ")\n"
        "```\n\n"
        "Composing the head INSIDE the outer feature extractor (the `module-composition` atom) "
        "yields a single Sequential-of-Sequentials with NO custom forward:\n\n"
        "```python\n"
        "classifier = nn.Sequential(\n"
        "    feature_extractor,    # outer Sequential of conv blocks.\n"
        "    head,                 # inner Sequential of Rearrange + Linear.\n"
        ")\n"
        "```\n\n"
        "**Why both atoms together.** `Rearrange` is what makes the head a `Sequential` — without "
        "it you'd need `forward()` to call `.flatten(1)`. `module-composition` is what makes the "
        "head a child of the outer classifier — without it you'd need a custom forward to call "
        "`self.features(x)` then `self.head(...)`. Combine them and the whole classifier is "
        "DECLARATIVE."
    ),
    "prompt_body": (
        "Implement `cx12_make_classifier(in_channels, num_classes)` — return an `nn.Sequential` "
        "with TWO children: a feature extractor and a classifier head.\n\n"
        "Layer composition:\n"
        "```\n"
        "outer = nn.Sequential(\n"
        "    feature_extractor,   # child 0: nn.Sequential of conv blocks.\n"
        "    head,                # child 1: nn.Sequential of Rearrange + Linear.\n"
        ")\n"
        "```\n\n"
        "Where:\n"
        "1. `feature_extractor = nn.Sequential(`\n"
        "       `nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, stride=2), nn.ReLU(),`\n"
        "       `nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2), nn.ReLU(),`\n"
        "   `)`  # (B, in_channels, 8, 8) -> (B, 16, 2, 2).\n"
        "2. `head = nn.Sequential(`\n"
        "       `Rearrange('b c h w -> b (c h w)'),`\n"
        "       `nn.Linear(16 * 2 * 2, num_classes),`\n"
        "   `)`  # (B, 16, 2, 2) -> (B, num_classes).\n"
        "3. Return `nn.Sequential(feature_extractor, head)`.\n\n"
        "Input shape: `(B, in_channels, 8, 8)`. Output shape: `(B, num_classes)`.\n\n"
        "Test checks:\n"
        "- Outer is `nn.Sequential` with EXACTLY 2 children, both themselves `nn.Sequential`.\n"
        "- Inner head's FIRST layer is a `Rearrange`.\n"
        "- Inner head's SECOND layer is `nn.Linear(64, num_classes)` (since 16*2*2 = 64).\n"
        "- End-to-end shape parity: `(B, in_channels, 8, 8) -> (B, num_classes)`.\n"
        "- Parameters from BOTH child Sequentials are collected by `outer.parameters()`."
    ),
    "stub_body": (
        "def cx12_make_classifier(in_channels: int, num_classes: int) -> 'nn.Sequential':\n"
        "    \"\"\"Return outer nn.Sequential of (feature_extractor, head) where head uses Rearrange.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "model = cx12_make_classifier(in_channels=3, num_classes=10)\n"
        "assert isinstance(model, nn.Sequential), f'outer must be nn.Sequential; got {type(model).__name__}'\n"
        "\n"
        "children = list(model.children())\n"
        "assert len(children) == 2, f'outer should have exactly 2 children; got {len(children)}'\n"
        "feature_extractor, head = children\n"
        "assert isinstance(feature_extractor, nn.Sequential), (\n"
        "    f'child 0 (feature_extractor) must be Sequential; got {type(feature_extractor).__name__}'\n"
        ")\n"
        "assert isinstance(head, nn.Sequential), (\n"
        "    f'child 1 (head) must be Sequential; got {type(head).__name__}'\n"
        ")\n"
        "\n"
        "# Case A: feature_extractor structure — 4 layers (2x conv + relu).\n"
        "fe_layers = list(feature_extractor.children())\n"
        "assert len(fe_layers) == 4, f'feature_extractor should have 4 layers; got {len(fe_layers)}'\n"
        "assert isinstance(fe_layers[0], nn.Conv2d) and fe_layers[0].in_channels == 3\n"
        "assert isinstance(fe_layers[2], nn.Conv2d) and fe_layers[2].out_channels == 16\n"
        "\n"
        "# Case B: head structure — Rearrange + Linear.\n"
        "head_layers = list(head.children())\n"
        "assert len(head_layers) == 2, f'head should have 2 layers; got {len(head_layers)}'\n"
        "assert isinstance(head_layers[0], Rearrange), (\n"
        "    f'head[0] must be einops Rearrange layer; got {type(head_layers[0]).__name__}'\n"
        ")\n"
        "assert isinstance(head_layers[1], nn.Linear), (\n"
        "    f'head[1] must be Linear; got {type(head_layers[1]).__name__}'\n"
        ")\n"
        "assert head_layers[1].in_features == 16 * 2 * 2, (\n"
        "    f'head Linear in_features should be 16*2*2=64; got {head_layers[1].in_features}'\n"
        ")\n"
        "assert head_layers[1].out_features == 10\n"
        "\n"
        "# Case C: end-to-end shape.\n"
        "for B in (1, 4):\n"
        "    x = t.randn(B, 3, 8, 8)\n"
        "    out = model(x)\n"
        "    assert out.shape == (B, 10), f'expected (B, num_classes)=({B}, 10); got {tuple(out.shape)}'\n"
        "\n"
        "# Case D: outer .parameters() recurses through BOTH children.\n"
        "all_params = list(model.parameters())\n"
        "fe_params = list(feature_extractor.parameters())\n"
        "head_params = list(head.parameters())\n"
        "assert len(all_params) == len(fe_params) + len(head_params), (\n"
        "    'outer .parameters() must collect from BOTH child Sequentials transitively'\n"
        ")\n"
        "# feature_extractor has 2 Conv2d (weight + bias each) = 4 params; head has 1 Linear = 2.\n"
        "assert len(all_params) == 6, f'expected 6 total params (4 fe + 2 head); got {len(all_params)}'\n"
        "\n"
        "# Case E: intermediate (after feature_extractor) shape proves the (B, 16, 2, 2) handoff.\n"
        "x = t.randn(2, 3, 8, 8)\n"
        "feat = feature_extractor(x)\n"
        "assert feat.shape == (2, 16, 2, 2), (\n"
        "    f'feature_extractor output should be (2, 16, 2, 2); got {tuple(feat.shape)}'\n"
        ")"
    ),
    "solution_body": (
        "def cx12_make_classifier(in_channels: int, num_classes: int):\n"
        "    # Atom A (module-composition): inner feature extractor as one child Sequential.\n"
        "    feature_extractor = nn.Sequential(\n"
        "        nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, stride=2),\n"
        "        nn.ReLU(),\n"
        "        nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2),\n"
        "        nn.ReLU(),\n"
        "    )\n"
        "    # Atom B (rearrange-as-sequential-layer): Rearrange + Linear as a Sequential head.\n"
        "    head = nn.Sequential(\n"
        "        Rearrange('b c h w -> b (c h w)'),\n"
        "        nn.Linear(16 * 2 * 2, num_classes),\n"
        "    )\n"
        "    # Atom A (module-composition): wrap them in an OUTER Sequential.\n"
        "    return nn.Sequential(feature_extractor, head)"
    ),
    "solution_notes": (
        "Sequential-of-Sequentials is fully legal and registers everything transitively — "
        "`outer.parameters()` walks into the nested Sequentials and finds every learnable tensor. "
        "The whole classifier is just a tree of Sequentials with a Rearrange leaf doing the "
        "flatten — no custom forward needed anywhere. This is the cleanest version of the "
        "'conv encoder + flatten + linear head' pattern."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["module-composition", "rearrange-as-sequential-layer"],
    "lo": (
        "Compose an outer nn.Sequential of two child Sequentials (feature extractor + head), "
        "where the head uses Rearrange-as-Sequential-layer to flatten before Linear, so the "
        "whole classifier is declarative with no custom forward."
    ),
}


SPECS = [spec_7, spec_8, spec_9, spec_10, spec_11, spec_12]


def main() -> None:
    for spec in SPECS:
        out = emit_composite(spec)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
