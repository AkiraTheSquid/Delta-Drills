"""Composite drills cx1..cx6 — batch-16 Q-cell (part0: einops / prereqs / numpy).

Six composite procedural drills exercising 2-3 atoms from the einops / numpy /
custom-tensor prereqs (ARENA part 0). Each composite forces the learner to
apply the atoms together in ONE function.

cx1  einops-rearrange + einops-rearrange-flatten            (flatten last two axes via grouped axis)
cx2  einops-repeat + einops-repeat-broadcast                (outer-product two grids via repeat + multiply)
cx3  einops-rearrange + einops-rearrange-flatten + tensor-wraps-ndarray
                                                            (reshape view on a wrapped-ndarray Tensor)
cx4  einops-rearrange-flatten + einops-repeat               (flatten then tile across new batch axis)
cx5  einops-repeat + einops-repeat-broadcast + broadcasting-rules
                                                            (outer-product (NR, NS) via repeat + broadcast)
cx6  einops-rearrange + einops-repeat                       (swap axes via rearrange, then repeat along the new dim)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_composite import emit_composite  # noqa: E402

INV = {a["atom_id"]: a for a in json.load(open("/tmp/drill_atoms.json"))}


def _subs(atom_ids):
    return [INV[a]["subtopic"] for a in atom_ids]


# ===========================================================================
# cx1 — flatten last two axes via grouped-axis rearrange "(h w) -> (h w)"
# ===========================================================================
spec_1 = {
    "atom_ids": ["einops-rearrange", "einops-rearrange-flatten"],
    "subtopics": _subs(["einops-rearrange", "einops-rearrange-flatten"]),
    "primary_atom": "einops-rearrange-flatten",
    "part": "part0",
    "exercise_index": 1,
    "exercise_title": "flatten last two axes via grouped-axis rearrange",
    "slug": "flatten-last-two-via-grouped-axis",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "`einops-rearrange` is the general pattern syntax `'src -> dst'`. "
        "`einops-rearrange-flatten` is the *grouped-axis* sub-pattern where you "
        "wrap names in parentheses to collapse them — `'b c h w -> b c (h w)'` "
        "flattens the spatial axes while preserving batch and channel.\n\n"
        "**The composition.** The grouped-axis flatten *is* a rearrange — there "
        "is no separate `flatten` op. The parens-grouping is the only thing that "
        "distinguishes a flatten-rearrange from an identity-rearrange. In ARENA "
        "code you reach for `'b c h w -> b c (h w)'` constantly: attention "
        "needs `(B, C, HW)` tokens, pooling needs `(B, C, HW)` to reduce, etc.\n\n"
        "**Inner-loop order matters.** `(h w)` flattens with `w` as the "
        "inner-fastest axis (matches row-major). `(w h)` would transpose first, "
        "then flatten — different bytes."
    ),
    "prompt_body": (
        "Build `cx1_flatten_grouped(x)` that takes a 4-D feature map "
        "`x` of shape `(B, C, H, W)` and returns a 3-D tensor of shape "
        "`(B, C, H*W)` using a SINGLE `rearrange` call with a grouped-axis "
        "pattern.\n\n"
        "Constraints:\n"
        "- Must use `einops.rearrange` (not `.view`, not `.reshape`, not "
        "`.flatten`).\n"
        "- Pattern must group `h` and `w` into `(h w)` — in that order, so the "
        "byte layout matches `x.reshape(B, C, H*W)`.\n"
        "- Batch and channel axes must remain in positions 0 and 1."
    ),
    "stub_body": (
        "def cx1_flatten_grouped(x):\n"
        '    """Flatten (B, C, H, W) -> (B, C, H*W) via grouped-axis rearrange."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# --- (a) basic shape + order ---\n"
        "x = t.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).float()\n"
        "y = cx1_flatten_grouped(x)\n"
        "assert y.shape == (2, 3, 20), f'expected (2,3,20), got {tuple(y.shape)}'\n"
        "# (h w) inner-loop-fastest order must match x.reshape(B, C, H*W) byte-for-byte.\n"
        "assert t.equal(y, x.reshape(2, 3, 20)), 'flatten order differs from row-major reshape'\n"
        "\n"
        "# --- (b) single batch, single channel ---\n"
        "x2 = t.arange(6).reshape(1, 1, 2, 3).float()\n"
        "y2 = cx1_flatten_grouped(x2)\n"
        "assert y2.shape == (1, 1, 6)\n"
        "assert t.equal(y2.squeeze(), t.tensor([0., 1., 2., 3., 4., 5.])), 'wrong flatten order'\n"
        "\n"
        "# --- (c) batch & channel axes preserved ---\n"
        "x3 = t.randn(7, 11, 3, 3)\n"
        "y3 = cx1_flatten_grouped(x3)\n"
        "assert y3.shape == (7, 11, 9)\n"
        "# Slice (b=4, c=2) — must match reshape of the same slice.\n"
        "assert t.allclose(y3[4, 2], x3[4, 2].reshape(9))"
    ),
    "solution_body": (
        "def cx1_flatten_grouped(x):\n"
        "    return rearrange(x, 'b c h w -> b c (h w)')"
    ),
    "solution_notes": (
        "Both atoms live in one expression: `rearrange(...)` is the rearrange "
        "atom, and the `(h w)` grouped-axis pattern is the flatten atom. Drop "
        "the parens and you get a shape error (`b c h w -> b c h w` is a "
        "different output). Reverse the order to `(w h)` and the byte layout "
        "differs from `x.reshape(B, C, H*W)` — test (a) catches it."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["rearrange-pattern-syntax", "rearrange-axis-composition-via-parens", "rearrange-flatten-preserves-batch"],
    "lo": "Compose einops.rearrange with a grouped-axis output to flatten the trailing two axes of a 4-D feature map.",
}


# ===========================================================================
# cx2 — outer-product two grids via repeat with new axis
# ===========================================================================
spec_2 = {
    "atom_ids": ["einops-repeat", "einops-repeat-broadcast"],
    "subtopics": _subs(["einops-repeat", "einops-repeat-broadcast"]),
    "primary_atom": "einops-repeat",
    "part": "part0",
    "exercise_index": 2,
    "exercise_title": "outer-product two 1-D grids via repeat + new-axis broadcast",
    "slug": "outer-product-grids-via-repeat",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "`einops-repeat` adds a new named axis to the output pattern and "
        "supplies its size via kwarg: `repeat(x, 'm -> m n', n=N)`. "
        "`einops-repeat-broadcast` is the *insight* that the inserted axis is "
        "a **stride-0 view** — no data is copied, einops just expands strides.\n\n"
        "**The composition for outer products.** To form an outer product "
        "`O[i, j] = u[i] * v[j]` you need both vectors expanded to a common "
        "`(M, N)` shape, then multiplied elementwise:\n"
        "1. `u: (M,)` becomes `(M, N)` via `repeat(u, 'm -> m n', n=N)`.\n"
        "2. `v: (N,)` becomes `(M, N)` via `repeat(v, 'n -> m n', m=M)`.\n"
        "3. Both expansions are stride-0 along the inserted axis, so the "
        "multiply is the only real work."
    ),
    "prompt_body": (
        "Build `cx2_outer_product(u, v)` that returns the outer product "
        "`O[i, j] = u[i] * v[j]` for 1-D tensors `u: (M,)` and `v: (N,)` using "
        "TWO `repeat` calls + one elementwise multiply.\n\n"
        "Constraints:\n"
        "- Must use `einops.repeat` for BOTH expansions (no `unsqueeze`, no "
        "`.expand`, no `torch.outer`).\n"
        "- The returned tensor must have shape `(M, N)` and values matching "
        "`torch.outer(u, v)`.\n"
        "- The two intermediate expansions must be stride-0 views (no copy) "
        "along the newly-inserted axis."
    ),
    "stub_body": (
        "def cx2_outer_product(u, v):\n"
        '    """Outer product via two repeat-broadcasts + elementwise multiply."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# --- (a) basic correctness against torch.outer ---\n"
        "u = t.tensor([1.0, 2.0, 3.0])\n"
        "v = t.tensor([10.0, 20.0, 30.0, 40.0])\n"
        "out = cx2_outer_product(u, v)\n"
        "assert out.shape == (3, 4), f'expected (3,4), got {tuple(out.shape)}'\n"
        "assert t.allclose(out, t.outer(u, v)), f'outer-product mismatch: {out}'\n"
        "\n"
        "# --- (b) random sanity ---\n"
        "u2 = t.randn(7)\n"
        "v2 = t.randn(5)\n"
        "out2 = cx2_outer_product(u2, v2)\n"
        "assert out2.shape == (7, 5)\n"
        "assert t.allclose(out2, t.outer(u2, v2), atol=1e-5)\n"
        "\n"
        "# --- (c) value spot-check ---\n"
        "u3 = t.tensor([2.0, 5.0])\n"
        "v3 = t.tensor([3.0, 7.0])\n"
        "out3 = cx2_outer_product(u3, v3)\n"
        "expected = t.tensor([[6.0, 14.0], [15.0, 35.0]])\n"
        "assert t.allclose(out3, expected), f'spot-check failed: {out3}'"
    ),
    "solution_body": (
        "def cx2_outer_product(u, v):\n"
        "    M, N = u.shape[0], v.shape[0]\n"
        "    U = repeat(u, 'm -> m n', n=N)   # (M, N), stride 0 along axis 1\n"
        "    V = repeat(v, 'n -> m n', m=M)   # (M, N), stride 0 along axis 0\n"
        "    return U * V"
    ),
    "solution_notes": (
        "Both atoms compose in one expression. `repeat(...)` with a new output "
        "axis is the repeat atom; the stride-0 nature of the inserted axis is "
        "the repeat-broadcast atom. The multiply is allowed to allocate (it "
        "must materialize the M*N output), but the two `repeat`s themselves "
        "do NOT copy — they emit broadcast views."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["repeat-add-axis", "repeat-inserts-zero-stride-axis", "repeat-pair-every-with-every"],
    "lo": "Use einops.repeat to expand two 1-D tensors into a shared (M, N) grid and form an outer product via elementwise multiply.",
}


# ===========================================================================
# cx3 — reshape view on a wrapped-ndarray Tensor (same storage, new view)
# ===========================================================================
spec_3 = {
    "atom_ids": ["einops-rearrange", "einops-rearrange-flatten", "tensor-wraps-ndarray"],
    "subtopics": _subs(["einops-rearrange", "einops-rearrange-flatten", "tensor-wraps-ndarray"]),
    "primary_atom": "tensor-wraps-ndarray",
    "part": "part0",
    "exercise_index": 3,
    "exercise_title": "rearrange-flatten on a wrapped-ndarray Tensor (no copy)",
    "slug": "rearrange-flatten-on-wrapped-ndarray-tensor",
    "atom_recap_md": (
        "## How these three atoms compose\n\n"
        "ARENA's custom autograd uses a thin `Tensor` wrapper around a numpy "
        "ndarray (the `array` attribute). The three atoms compose like so:\n\n"
        "1. **`tensor-wraps-ndarray`** — `Tensor.__init__` stores the ndarray "
        "directly. Reshapes inside `Tensor` ops should reuse the underlying "
        "buffer (no copy), so the wrapped `Tensor` is just a structured view.\n"
        "2. **`einops-rearrange`** — the general rearrange pattern syntax. "
        "Works on any array-like, including raw ndarrays.\n"
        "3. **`einops-rearrange-flatten`** — the `'b c h w -> b (c h w)'` "
        "grouped-axis flatten — the same op as cx1, but applied to the inner "
        "ndarray of a wrapped `Tensor` and wrapped back up.\n\n"
        "**The composition.** Build a `Tensor.flatten_bchw()` method that "
        "calls `rearrange` on `self.array` (an ndarray), wraps the result back "
        "into a new `Tensor`, and the two wrappers SHARE storage."
    ),
    "prompt_body": (
        "A minimal `Tensor` class is provided at module scope (see the stub). "
        "It wraps a single numpy ndarray as `self.array`.\n\n"
        "Build `cx3_flatten_bchw(ten)` that takes a `Tensor` whose `.array` "
        "has shape `(B, C, H, W)` and returns a NEW `Tensor` whose `.array` "
        "has shape `(B, C*H*W)`.\n\n"
        "Constraints:\n"
        "- Must use `einops.rearrange` on the inner ndarray with the "
        "`'b c h w -> b (c h w)'` pattern.\n"
        "- The output `Tensor`'s `.array` MUST share memory with the input "
        "`Tensor`'s `.array` (no copy) — verified via "
        "`np.shares_memory(...)`.\n"
        "- Mutating the input ndarray must be visible through the output."
    ),
    "stub_body": (
        "class Tensor:\n"
        '    """Minimal ARENA-style Tensor: wraps a single ndarray, no metadata."""\n'
        "    def __init__(self, array):\n"
        "        assert isinstance(array, np.ndarray), 'Tensor wraps an ndarray'\n"
        "        self.array = array\n"
        "\n"
        "    def __repr__(self):\n"
        "        return f'Tensor(shape={self.array.shape}, dtype={self.array.dtype})'\n"
        "\n"
        "def cx3_flatten_bchw(ten):\n"
        '    """Flatten (B, C, H, W) -> (B, C*H*W) on a wrapped Tensor, no copy."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# --- (a) shape correct ---\n"
        "arr = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)\n"
        "ten = Tensor(arr)\n"
        "out = cx3_flatten_bchw(ten)\n"
        "assert isinstance(out, Tensor), f'must return a Tensor, got {type(out)}'\n"
        "assert out.array.shape == (2, 60), f'expected (2,60), got {out.array.shape}'\n"
        "\n"
        "# --- (b) values match (c h w) inner-loop-fastest order ---\n"
        "expected = arr.reshape(2, 60)\n"
        "assert np.array_equal(out.array, expected), 'flatten order differs from row-major reshape'\n"
        "\n"
        "# --- (c) NO COPY: output ndarray shares memory with input ndarray ---\n"
        "assert np.shares_memory(out.array, ten.array), (\n"
        "    'output array must share memory with the wrapped input — '\n"
        "    'did you accidentally call .copy() or do an out-of-place op?')\n"
        "\n"
        "# --- (d) mutation through the wrapper is visible in the flattened view ---\n"
        "ten.array[0, 0, 0, 0] = -999.0\n"
        "assert out.array[0, 0] == -999.0, (\n"
        "    'mutating input.array[0,0,0,0] should be visible at output.array[0,0] '\n"
        "    'since they share storage')\n"
        "\n"
        "# --- (e) different shape ---\n"
        "arr2 = np.random.randn(1, 2, 3, 3).astype(np.float32)\n"
        "ten2 = Tensor(arr2)\n"
        "out2 = cx3_flatten_bchw(ten2)\n"
        "assert out2.array.shape == (1, 18)\n"
        "assert np.allclose(out2.array, arr2.reshape(1, 18))\n"
        "assert np.shares_memory(out2.array, ten2.array)"
    ),
    "solution_body": (
        "def cx3_flatten_bchw(ten):\n"
        "    flat = rearrange(ten.array, 'b c h w -> b (c h w)')\n"
        "    # einops on a contiguous ndarray returns a reshape view — same buffer.\n"
        "    return Tensor(flat)"
    ),
    "solution_notes": (
        "All three atoms compose into a four-line function: the `Tensor(...)` "
        "wrapper (tensor-wraps-ndarray) holds an ndarray; `rearrange(...)` "
        "(einops-rearrange) is the call; the `(c h w)` grouped-axis "
        "(einops-rearrange-flatten) is the pattern. Calling `.copy()` or "
        "swapping in `np.asarray(...).reshape(...).copy()` breaks the "
        "`shares_memory` check in test (c)."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["tensor-wraps-ndarray", "rearrange-axis-composition-via-parens", "rearrange-flatten-preserves-batch"],
    "lo": "Apply einops.rearrange with a grouped-axis flatten to the inner ndarray of a wrapped Tensor, returning a same-storage view wrapped back into a new Tensor.",
}


# ===========================================================================
# cx4 — flatten then tile across new batch axis
# ===========================================================================
spec_4 = {
    "atom_ids": ["einops-rearrange-flatten", "einops-repeat"],
    "subtopics": _subs(["einops-rearrange-flatten", "einops-repeat"]),
    "primary_atom": "einops-rearrange-flatten",
    "part": "part0",
    "exercise_index": 4,
    "exercise_title": "flatten a CHW image then tile across a new batch axis",
    "slug": "flatten-then-tile-across-batch",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "`einops-rearrange-flatten` collapses named axes via parens: "
        "`'c h w -> (c h w)'` flattens a 3-D feature map into a 1-D vector. "
        "`einops-repeat` then inserts a new named axis: "
        "`'features -> b features', b=B` tiles the same flat vector across a "
        "new batch dim.\n\n"
        "**Why this pattern shows up.** When you have a single template image "
        "(e.g. a CNN's input mean or a positional embedding) and need to "
        "broadcast it as a flat feature vector across a whole training batch, "
        "you flatten first (so downstream Linear layers see `(B, D)`) and "
        "then `repeat` to insert `B`.\n\n"
        "**Order matters.** Flatten first, repeat second — repeating BEFORE "
        "flatten gives a `(B, C, H, W)` tensor which the Linear head can't "
        "consume without another flatten."
    ),
    "prompt_body": (
        "Build `cx4_flatten_then_tile(x, batch)` that takes a 3-D image-like "
        "tensor `x` of shape `(C, H, W)` and tiles it across `batch` rows of a "
        "new batch axis, producing shape `(batch, C*H*W)`.\n\n"
        "Constraints:\n"
        "- Use ONE `rearrange` to flatten `(C, H, W) -> (C*H*W,)`.\n"
        "- Use ONE `repeat` to add the batch axis: `'features -> b features'`.\n"
        "- The flat row at index `i` of the output must equal "
        "`x.reshape(C*H*W)` for every `i`.\n"
        "- No `.unsqueeze().expand()`, no `torch.stack`, no manual loops."
    ),
    "stub_body": (
        "def cx4_flatten_then_tile(x, batch):\n"
        '    """Flatten (C,H,W) then repeat to (batch, C*H*W)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# --- (a) basic shape ---\n"
        "x = t.arange(2 * 3 * 4).reshape(2, 3, 4).float()\n"
        "out = cx4_flatten_then_tile(x, batch=5)\n"
        "assert out.shape == (5, 24), f'expected (5, 24), got {tuple(out.shape)}'\n"
        "\n"
        "# --- (b) every row equals the flat reference vector ---\n"
        "flat_ref = x.reshape(24)\n"
        "for b in range(5):\n"
        "    assert t.equal(out[b], flat_ref), f'row {b} differs from flat reference'\n"
        "\n"
        "# --- (c) inner-loop order is (c h w), matches reshape ---\n"
        "x2 = t.tensor([[[0., 1.], [2., 3.]],\n"
        "               [[4., 5.], [6., 7.]]])  # (C=2, H=2, W=2)\n"
        "out2 = cx4_flatten_then_tile(x2, batch=3)\n"
        "assert out2.shape == (3, 8)\n"
        "assert t.equal(out2[0], t.tensor([0., 1., 2., 3., 4., 5., 6., 7.]))\n"
        "\n"
        "# --- (d) batch=1 still works ---\n"
        "out3 = cx4_flatten_then_tile(x, batch=1)\n"
        "assert out3.shape == (1, 24)\n"
        "assert t.equal(out3[0], flat_ref)"
    ),
    "solution_body": (
        "def cx4_flatten_then_tile(x, batch):\n"
        "    flat = rearrange(x, 'c h w -> (c h w)')        # flatten atom\n"
        "    return repeat(flat, 'd -> b d', b=batch)        # repeat atom adds B"
    ),
    "solution_notes": (
        "Two atoms, two lines. The flatten is `'c h w -> (c h w)'` (grouped "
        "axis collapses to a single 1-D dim); the tile is `'d -> b d', b=B` "
        "(new axis introduced by repeat). Doing the steps in the wrong order "
        "(tile-then-flatten) gives `(batch, C*H*W)` but only via a second "
        "flatten step — and a stricter test would catch the intermediate "
        "shape."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["rearrange-axis-composition-via-parens", "repeat-add-axis"],
    "lo": "Flatten a CHW image with einops.rearrange grouped-axis, then einops.repeat to broadcast across a new batch dimension.",
}


# ===========================================================================
# cx5 — outer-product (NR, NS) pattern via repeat + broadcast
# ===========================================================================
spec_5 = {
    "atom_ids": ["einops-repeat", "einops-repeat-broadcast", "broadcasting-rules"],
    "subtopics": _subs(["einops-repeat", "einops-repeat-broadcast", "broadcasting-rules"]),
    "primary_atom": "einops-repeat-broadcast",
    "part": "part0",
    "exercise_index": 5,
    "exercise_title": "every-ray-with-every-screen-pixel pairing via repeat + broadcast",
    "slug": "outer-product-nr-ns-via-repeat-broadcast",
    "atom_recap_md": (
        "## How these three atoms compose\n\n"
        "ARENA's ray-tracing chapter needs to pair every ray with every screen "
        "pixel (or every triangle) without materialising the cross-product. "
        "The standard pattern combines three atoms:\n\n"
        "1. **`einops-repeat`** — inserts the missing axis via "
        "`repeat(x, 'nr d -> nr ns d', ns=NS)`. The output is `(NR, NS, D)`.\n"
        "2. **`einops-repeat-broadcast`** — the insight that the inserted "
        "axis has **stride 0**, so this is a broadcast VIEW, not a copy. "
        "`data_ptr()` of the output matches the input.\n"
        "3. **`broadcasting-rules`** — right-aligned shape arithmetic: "
        "`(NR, 1, D)` and `(1, NS, D)` broadcast to `(NR, NS, D)` "
        "elementwise.\n\n"
        "**The composition.** Two `repeat` calls produce stride-0 broadcast "
        "views with matching shapes, then any elementwise op (subtract, "
        "multiply, ...) follows numpy broadcasting rules and produces "
        "`(NR, NS, D)` outputs."
    ),
    "prompt_body": (
        "Build `cx5_pair_rays_screens(rays, screens)` that takes:\n"
        "- `rays`: shape `(NR, 3)` — `NR` direction vectors\n"
        "- `screens`: shape `(NS, 3)` — `NS` screen-pixel positions\n\n"
        "and returns the elementwise difference `screens[s] - rays[r]` for "
        "every `(r, s)` pair, as a tensor of shape `(NR, NS, 3)`.\n\n"
        "Constraints:\n"
        "- Must use `einops.repeat` to broadcast BOTH inputs to shape "
        "`(NR, NS, 3)` first.\n"
        "- The two intermediate broadcast tensors must be stride-0 views "
        "(no copy) — verified via `.data_ptr()`.\n"
        "- The subtract is allowed to allocate (it produces the output).\n"
        "- No `.unsqueeze().expand()`, no `torch.broadcast_to`."
    ),
    "stub_body": (
        "def cx5_pair_rays_screens(rays, screens):\n"
        '    """For every (ray, screen) pair, return screens - rays as (NR, NS, 3)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# --- (a) shape + correctness against a naive loop ---\n"
        "NR, NS = 4, 5\n"
        "rays = t.randn(NR, 3)\n"
        "screens = t.randn(NS, 3)\n"
        "out = cx5_pair_rays_screens(rays, screens)\n"
        "assert out.shape == (NR, NS, 3), f'expected (NR, NS, 3), got {tuple(out.shape)}'\n"
        "for r in range(NR):\n"
        "    for s in range(NS):\n"
        "        assert t.allclose(out[r, s], screens[s] - rays[r]), f'mismatch at ({r},{s})'\n"
        "\n"
        "# --- (b) hand-built spot-check ---\n"
        "rays_h = t.tensor([[1.0, 0.0, 0.0],\n"
        "                   [0.0, 1.0, 0.0]])      # NR=2\n"
        "screens_h = t.tensor([[10.0, 10.0, 10.0],\n"
        "                      [20.0, 20.0, 20.0],\n"
        "                      [30.0, 30.0, 30.0]]) # NS=3\n"
        "out_h = cx5_pair_rays_screens(rays_h, screens_h)\n"
        "assert out_h.shape == (2, 3, 3)\n"
        "# ray 0 vs screen 0: (10-1, 10-0, 10-0) = (9, 10, 10)\n"
        "assert t.allclose(out_h[0, 0], t.tensor([9.0, 10.0, 10.0]))\n"
        "# ray 1 vs screen 2: (30-0, 30-1, 30-0) = (30, 29, 30)\n"
        "assert t.allclose(out_h[1, 2], t.tensor([30.0, 29.0, 30.0]))\n"
        "\n"
        "# --- (c) scale: no allocation blowup ---\n"
        "big_rays = t.randn(2000, 3)\n"
        "big_screens = t.randn(100, 3)\n"
        "big_out = cx5_pair_rays_screens(big_rays, big_screens)\n"
        "assert big_out.shape == (2000, 100, 3)"
    ),
    "solution_body": (
        "def cx5_pair_rays_screens(rays, screens):\n"
        "    NR, NS = rays.shape[0], screens.shape[0]\n"
        "    # repeat inserts a stride-0 axis (the repeat-broadcast atom).\n"
        "    rays_b    = repeat(rays,    'nr d -> nr ns d', ns=NS)   # (NR, NS, 3)\n"
        "    screens_b = repeat(screens, 'ns d -> nr ns d', nr=NR)   # (NR, NS, 3)\n"
        "    # Now broadcasting-rules — both shapes are identical, elementwise sub works.\n"
        "    return screens_b - rays_b"
    ),
    "solution_notes": (
        "Three atoms in three lines. `repeat(...)` (einops-repeat) is the "
        "call; the stride-0 nature of the inserted `ns` / `nr` axis is the "
        "repeat-broadcast atom; the final elementwise subtract obeys numpy "
        "broadcasting rules — both intermediates have shape `(NR, NS, 3)`, "
        "so broadcasting is trivial and the output is `(NR, NS, 3)`."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["repeat-add-axis", "repeat-inserts-zero-stride-axis", "repeat-pair-every-with-every", "broadcasting-right-align"],
    "lo": "Apply einops.repeat twice to materialize a stride-0 (NR, NS, D) broadcast view of two 1-D-stacked tensors, then use numpy broadcasting rules for the elementwise op.",
}


# ===========================================================================
# cx6 — swap axes via rearrange, then repeat along the new dim
# ===========================================================================
spec_6 = {
    "atom_ids": ["einops-rearrange", "einops-repeat"],
    "subtopics": _subs(["einops-rearrange", "einops-repeat"]),
    "primary_atom": "einops-rearrange",
    "part": "part0",
    "exercise_index": 6,
    "exercise_title": "transpose with rearrange, then repeat along a new axis",
    "slug": "transpose-then-repeat-new-axis",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "`einops-rearrange` does pure axis reordering / regrouping — the data "
        "doesn't change, only how axes are named. `'h w -> w h'` is a "
        "transpose. `einops-repeat` then introduces a new size that wasn't in "
        "the input, supplied as a kwarg.\n\n"
        "**The composition.** Transpose `(H, W) -> (W, H)`, then tile across "
        "a new leading channel axis `(C, W, H)` — useful when you have a "
        "single greyscale image laid out `(H, W)` and need to feed a "
        "(C, H', W') CNN that expects channel-first AND the spatial order "
        "swapped (e.g. an x/y flip)."
    ),
    "prompt_body": (
        "Build `cx6_transpose_then_channelize(x, channels)`.\n\n"
        "Input: `x` of shape `(H, W)`. Output: a tensor of shape "
        "`(channels, W, H)` where:\n"
        "- The spatial axes are swapped (transposed `H` and `W`).\n"
        "- The new leading axis tiles `channels` copies of the transposed "
        "image.\n\n"
        "Constraints:\n"
        "- Step 1: ONE `rearrange` call to perform the `(H, W) -> (W, H)` "
        "transpose.\n"
        "- Step 2: ONE `repeat` call to add the leading `channels` axis, "
        "producing `(channels, W, H)`.\n"
        "- Every channel slice along axis 0 must equal the transposed image.\n"
        "- No `.t()`, no `.transpose()`, no `.expand()`, no `torch.stack`."
    ),
    "stub_body": (
        "def cx6_transpose_then_channelize(x, channels):\n"
        '    """Transpose (H,W)->(W,H) via rearrange, then repeat to (C, W, H)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# --- (a) shape + value correctness ---\n"
        "x = t.arange(12).reshape(3, 4).float()  # (H=3, W=4)\n"
        "out = cx6_transpose_then_channelize(x, channels=2)\n"
        "assert out.shape == (2, 4, 3), f'expected (2, 4, 3), got {tuple(out.shape)}'\n"
        "expected_t = x.t()  # reference transpose, shape (4, 3)\n"
        "for c in range(2):\n"
        "    assert t.equal(out[c], expected_t), f'channel {c} != transposed input'\n"
        "\n"
        "# --- (b) spot-check transpose semantics ---\n"
        "x2 = t.tensor([[1.0, 2.0, 3.0],\n"
        "               [4.0, 5.0, 6.0]])       # (H=2, W=3)\n"
        "out2 = cx6_transpose_then_channelize(x2, channels=4)\n"
        "assert out2.shape == (4, 3, 2)\n"
        "# out2[any_c, w, h] == x2[h, w]\n"
        "assert out2[0, 0, 0] == 1.0    # h=0, w=0\n"
        "assert out2[0, 1, 0] == 2.0    # h=0, w=1\n"
        "assert out2[0, 2, 1] == 6.0    # h=1, w=2\n"
        "assert out2[3, 2, 1] == 6.0    # also for channel 3\n"
        "\n"
        "# --- (c) channels=1 still works ---\n"
        "out3 = cx6_transpose_then_channelize(x, channels=1)\n"
        "assert out3.shape == (1, 4, 3)\n"
        "assert t.equal(out3[0], x.t())"
    ),
    "solution_body": (
        "def cx6_transpose_then_channelize(x, channels):\n"
        "    swapped = rearrange(x, 'h w -> w h')                  # rearrange atom\n"
        "    return repeat(swapped, 'w h -> c w h', c=channels)    # repeat atom"
    ),
    "solution_notes": (
        "Two atoms, two calls. The transpose is the canonical `rearrange` use "
        "(reordering output axes); the leading-channel tile is the canonical "
        "`repeat` use (new axis introduced by kwarg). Swap the order "
        "(repeat-then-rearrange) and the rearrange pattern needs three axes "
        "instead of two — both work but the assigned shape contract is "
        "transpose-first."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["rearrange-pattern-syntax", "repeat-add-axis"],
    "lo": "Use einops.rearrange to swap two spatial axes of an HxW image and einops.repeat to tile the result along a new channel axis.",
}


# ---------------------------------------------------------------------------
# Emit all six.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for spec in (spec_1, spec_2, spec_3, spec_4, spec_5, spec_6):
        path = emit_composite(spec)
        print(f"wrote {path}")
