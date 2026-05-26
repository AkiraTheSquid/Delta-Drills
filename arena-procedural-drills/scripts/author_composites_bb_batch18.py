"""Composite drills cx7..cx12 — batch-18 (BB-cell, part2 ARENA CNNs).

Six composite procedural drills exercising 2-atom pairs from the ARENA CNN
prereq atoms. Each drill picks an anchor + neighbour pair that wires together
in the from-scratch conv1d/conv2d implementations in ARENA part 2.

cx7   conv-padding-zero       + conv-stride-downsample      (pad + stride combined)
cx8   as-strided-windowing    + conv-padding-zero           (pad first, then strided window)
cx9   as-strided-windowing    + conv-stride-downsample      (strided window with stride > 1)
cx10  conv-output-shape       + conv-padding-zero           (output shape includes padding)
cx11  conv-output-shape       + conv-stride-downsample      (output shape with stride downsample)
cx12  as-strided-windowing    + conv-windowing-1d           (1D conv via 1D windowed view)
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
# cx7 — apply zero-padding AND stride downsample to predict + execute a 1-D conv
# ===========================================================================
spec_7 = {
    "atom_ids": ["conv-padding-zero", "conv-stride-downsample"],
    "subtopics": _subs(["conv-padding-zero", "conv-stride-downsample"]),
    "primary_atom": "conv-padding-zero",
    "part": "part2",
    "exercise_index": 7,
    "exercise_title": "conv1d with both zero padding and stride > 1",
    "slug": "pad-then-stride-conv1d",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "ARENA's from-scratch `conv1d` separates the conv into two prep steps before the einsum:\n"
        "1. **Zero-pad** the input on both sides with `padding=P` zeros (the `conv-padding-zero` atom). "
        "After this, the effective input length becomes `W + 2*P`.\n"
        "2. **Stride-downsample** when extracting windows: the number of windows over the padded input "
        "with stride `S` is `OW = (W + 2*P - K) // S + 1` (the `conv-stride-downsample` atom — note "
        "the `+1` for the leading window).\n\n"
        "The composition: pad first, THEN apply the strided output-length formula on top of the padded "
        "length. This drill exercises both in one function: pad the input, return the padded tensor "
        "AND the predicted strided output length.\n\n"
        "**Anatomy.**\n"
        "- `x_pad = x.new_zeros(B, IC, W + 2*P); x_pad[..., P:P+W] = x` — atom A.\n"
        "- `OW = (W + 2*P - K) // S + 1` — atom B applied to the PADDED length.\n\n"
        "**Why this matters.** ResNet-style 'stride-2 + same-padding' downsampling relies on this exact "
        "composition: the same-padding term cancels the floor-division by-one error so the output is "
        "*exactly* `W // S`. Forget the `2*P` term and you get the canonical off-by-one bug."
    ),
    "prompt_body": (
        "Implement `cx7_pad_then_strided_outlen(x, K, S, P)`.\n\n"
        "- `x`: float tensor of shape `(B, IC, W)`.\n"
        "- `K`: kernel width (int).\n"
        "- `S`: stride (int, >= 1).\n"
        "- `P`: padding (int, >= 0). Same amount on left and right.\n\n"
        "Return `(x_padded, OW)`:\n"
        "- `x_padded`: shape `(B, IC, W + 2*P)`. Interior `[P : P+W]` equals `x`; left/right `P` "
        "columns are exactly zero.\n"
        "- `OW`: integer output width of a stride-`S` conv with kernel `K` over the padded input.\n\n"
        "1. **Pad** — allocate a zero buffer of shape `(B, IC, W + 2*P)` via `x.new_zeros(...)` so the "
        "dtype/device track `x`. Slice-assign `x` into columns `[P : P+W]`.\n"
        "2. **Strided output length** — apply `OW = (W + 2*P - K) // S + 1` to the PADDED length. "
        "Note this includes the `+1` for the leading window."
    ),
    "stub_body": (
        "def cx7_pad_then_strided_outlen(x, K, S, P):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "from torch.nn import functional as F\n"
        "\n"
        "# Case A: hand-built — pad=1, stride=2, K=3 on a length-5 input.\n"
        "x = t.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])  # (1, 1, 5)\n"
        "x_pad, OW = cx7_pad_then_strided_outlen(x, K=3, S=2, P=1)\n"
        "assert tuple(x_pad.shape) == (1, 1, 7), f'expected (1,1,7), got {tuple(x_pad.shape)}'\n"
        "assert OW == 3, f'(5+2-3)//2 + 1 = 3, got {OW}'\n"
        "# Boundary cells exactly 0.\n"
        "assert (x_pad[..., :1] == 0).all(), 'left pad must be exact 0'\n"
        "assert (x_pad[..., -1:] == 0).all(), 'right pad must be exact 0'\n"
        "assert t.allclose(x_pad[..., 1:6], x), 'interior must equal x'\n"
        "\n"
        "# Case B: P=0 — pad is a no-op, OW falls back to the unpadded formula.\n"
        "x = t.randn(2, 3, 10)\n"
        "x_pad, OW = cx7_pad_then_strided_outlen(x, K=3, S=2, P=0)\n"
        "assert tuple(x_pad.shape) == (2, 3, 10)\n"
        "assert t.allclose(x_pad, x), 'P=0 should leave x unchanged'\n"
        "assert OW == (10 - 3) // 2 + 1  # = 4\n"
        "\n"
        "# Case C: ResNet-style 'same-pad + stride-2' yields exactly W//S.\n"
        "# K=3, P=1, S=2: OW = (W + 2 - 3)//2 + 1 = (W-1)//2 + 1 = W//2 for even W.\n"
        "for W in [8, 16, 32, 64]:\n"
        "    x = t.zeros(1, 1, W)\n"
        "    _, OW = cx7_pad_then_strided_outlen(x, K=3, S=2, P=1)\n"
        "    assert OW == W // 2, f'same-pad+stride-2: W={W} OW={OW}, expected {W//2}'\n"
        "\n"
        "# Case D: cross-check against F.conv1d for several configs.\n"
        "rng = t.Generator().manual_seed(7)\n"
        "for B, IC, W, K, S, P in [\n"
        "    (1, 1, 10, 3, 1, 1),\n"
        "    (2, 3, 12, 5, 2, 2),\n"
        "    (1, 4,  9, 3, 3, 0),\n"
        "    (3, 1, 20, 4, 2, 1),\n"
        "]:\n"
        "    x = t.randn(B, IC, W, generator=rng)\n"
        "    x_pad, OW = cx7_pad_then_strided_outlen(x, K, S, P)\n"
        "    assert tuple(x_pad.shape) == (B, IC, W + 2*P)\n"
        "    # F.conv1d sanity: a random kernel must produce OW columns.\n"
        "    OC = 2\n"
        "    w = t.randn(OC, IC, K, generator=rng)\n"
        "    y = F.conv1d(x, w, stride=S, padding=P)\n"
        "    assert y.shape[-1] == OW, (\n"
        "        f'OW mismatch: predicted {OW}, F.conv1d gave {y.shape[-1]} for '\n"
        "        f'B={B} IC={IC} W={W} K={K} S={S} P={P}'\n"
        "    )\n"
        "\n"
        "# Case E: off-by-one trap — verify we did NOT just return W // S.\n"
        "x = t.zeros(1, 1, 32)\n"
        "_, OW = cx7_pad_then_strided_outlen(x, K=3, S=2, P=0)\n"
        "assert OW == 15, f'(32-3)//2 + 1 = 15 (NOT 16!), got {OW}'"
    ),
    "solution_body": (
        "def cx7_pad_then_strided_outlen(x, K, S, P):\n"
        "    B, IC, W = x.shape\n"
        "    # Atom A (conv-padding-zero): allocate a zero buffer, slice-assign the interior.\n"
        "    x_pad = x.new_zeros(B, IC, W + 2 * P)\n"
        "    x_pad[..., P : P + W] = x\n"
        "    # Atom B (conv-stride-downsample): formula applied to the PADDED length.\n"
        "    # Note the +1 — leading window starts at index 0.\n"
        "    OW = (W + 2 * P - K) // S + 1\n"
        "    return x_pad, OW"
    ),
    "solution_notes": (
        "The two atoms compose in series: pad expands the effective input by `2*P`, then the strided "
        "output formula consumes that expanded length. Forgetting the `+1` or the `2*P` term is the "
        "canonical conv-shape bug. The same-pad + stride-2 case (Case C) is why ResNets work cleanly: "
        "with `K=3, P=1, S=2` the formula collapses to `W // 2` for even `W`."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["conv-padding-zero", "conv-stride-downsample"],
    "lo": (
        "Compose zero-padding (allocate zero buffer + slice-assign interior) with stride-downsample "
        "arithmetic ((W + 2P - K)//S + 1 applied to the padded length) to predict and prepare the "
        "1-D input for a strided, padded conv."
    ),
}


# ===========================================================================
# cx8 — pad input, then build the strided window view via as_strided
# ===========================================================================
spec_8 = {
    "atom_ids": ["as-strided-windowing", "conv-padding-zero"],
    "subtopics": _subs(["as-strided-windowing", "conv-padding-zero"]),
    "primary_atom": "as-strided-windowing",
    "part": "part2",
    "exercise_index": 8,
    "exercise_title": "pad first, then build the as_strided window view",
    "slug": "pad-then-as-strided-windows",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "ARENA's `conv1d_minimal` with padding splits the work into two zero-copy-friendly stages:\n"
        "1. **Pad** the input with zeros on both ends (`conv-padding-zero`). This is a materializing "
        "op — it allocates a new tensor `x_pad` of shape `(B, IC, W + 2*P)`.\n"
        "2. **Window via as_strided** (`as-strided-windowing`). Read `x_pad.stride()` and build a "
        "`(B, IC, OW, K)` view onto the PADDED tensor. No copy here — the view aliases `x_pad`'s "
        "storage.\n\n"
        "**Why ordering matters.** You must read `x_pad.stride()`, NOT `x.stride()`. After padding, "
        "the storage and strides have changed (you're looking at a new contiguous tensor sized "
        "`W + 2*P`). Hard-coding `x.stride()` is one of the canonical ARENA bugs — your windows then "
        "read the wrong memory cells.\n\n"
        "**Anatomy.**\n"
        "- `x_pad = x.new_zeros(B, IC, W + 2*P); x_pad[..., P:P+W] = x` — atom B.\n"
        "- `s_b, s_ic, s_w = x_pad.stride()` — read the PADDED tensor's strides.\n"
        "- `OW = W + 2*P - K + 1` — stride-1 output length on the padded input.\n"
        "- `win = x_pad.as_strided(size=(B, IC, OW, K), stride=(s_b, s_ic, s_w, s_w))` — atom A.\n\n"
        "**Result.** `win` is a stride-1 windowed view of the padded input. Contract against a "
        "`(OC, IC, K)` kernel via `einops.einsum` to get the conv output."
    ),
    "prompt_body": (
        "Implement `cx8_padded_windows(x, K, P)`.\n\n"
        "- `x`: float tensor of shape `(B, IC, W)`.\n"
        "- `K`: kernel width.\n"
        "- `P`: padding amount on each side (>= 0).\n\n"
        "Return `(x_padded, win)`:\n"
        "- `x_padded`: shape `(B, IC, W + 2*P)`. Standard zero-pad.\n"
        "- `win`: shape `(B, IC, OW, K)` where `OW = W + 2*P - K + 1` — a stride-1 windowed view "
        "onto `x_padded`. Must share storage with `x_padded` (no copy).\n\n"
        "1. **Pad** — use `x.new_zeros(...)` + slice assignment.\n"
        "2. **Read PADDED strides** — `s_b, s_ic, s_w = x_padded.stride()`. Do NOT use `x.stride()` "
        "— after padding the strides are different.\n"
        "3. **As-strided window** — `x_padded.as_strided(size=(B, IC, OW, K), stride=(s_b, s_ic, s_w, "
        "s_w))`. The trailing `(s_w, s_w)` pair is the stride-1 windowing pattern.\n\n"
        "The test verifies the view is correct AND that `win.data_ptr() == x_padded.data_ptr()` "
        "(no copy), AND cross-checks against `F.conv1d(x, weight, padding=P)` after einsum-ing the "
        "windows against a random kernel."
    ),
    "stub_body": (
        "def cx8_padded_windows(x, K, P):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "from torch.nn import functional as F\n"
        "\n"
        "# Case A: hand-built — verify pad + window shape.\n"
        "x = t.arange(1.0, 6.0).reshape(1, 1, 5).contiguous()  # [1,2,3,4,5]\n"
        "x_pad, win = cx8_padded_windows(x, K=3, P=1)\n"
        "assert tuple(x_pad.shape) == (1, 1, 7), f'pad shape: {tuple(x_pad.shape)}'\n"
        "assert tuple(win.shape) == (1, 1, 5, 3), f'win shape: {tuple(win.shape)}'  # OW = 7-3+1 = 5\n"
        "# Window 0 over padded = [0, 1, 2].\n"
        "assert t.allclose(win[0, 0, 0], t.tensor([0.0, 1.0, 2.0])), f'window 0: {win[0,0,0]}'\n"
        "# Window 1 = [1, 2, 3].\n"
        "assert t.allclose(win[0, 0, 1], t.tensor([1.0, 2.0, 3.0]))\n"
        "# Last window = [4, 5, 0] (catches trailing pad cell).\n"
        "assert t.allclose(win[0, 0, 4], t.tensor([4.0, 5.0, 0.0])), f'last window: {win[0,0,4]}'\n"
        "\n"
        "# Case B: no-copy property — win must alias x_pad's storage.\n"
        "assert win.data_ptr() == x_pad.data_ptr(), 'windows must be a view of x_pad (share storage)'\n"
        "\n"
        "# Case C: stride-read-from-padded sanity. If a solution reads x.stride() instead of\n"
        "# x_pad.stride(), the windowing breaks for non-contiguous x. Verify the contiguous case\n"
        "# still works (lets us cross-check via F.conv1d).\n"
        "rng = t.Generator().manual_seed(8)\n"
        "B, IC, W, K, P, OC = 2, 3, 12, 5, 2, 4\n"
        "x = t.randn(B, IC, W, generator=rng)\n"
        "weight = t.randn(OC, IC, K, generator=rng)\n"
        "x_pad, win = cx8_padded_windows(x, K, P)\n"
        "OW = W + 2 * P - K + 1\n"
        "assert tuple(win.shape) == (B, IC, OW, K)\n"
        "y_manual = einops.einsum(win, weight, 'b ic ow kw, oc ic kw -> b oc ow')\n"
        "y_native = F.conv1d(x, weight, padding=P)\n"
        "assert t.allclose(y_manual, y_native, atol=1e-4), 'einsum(windows, weight) must equal F.conv1d'\n"
        "\n"
        "# Case D: P=0 — degenerates to plain windowing on the un-padded input.\n"
        "x = t.arange(1.0, 11.0).reshape(1, 1, 10).contiguous()\n"
        "x_pad, win = cx8_padded_windows(x, K=3, P=0)\n"
        "assert tuple(x_pad.shape) == (1, 1, 10)\n"
        "assert t.allclose(x_pad, x)\n"
        "assert tuple(win.shape) == (1, 1, 8, 3)  # OW = 10-3+1 = 8\n"
        "for k in range(8):\n"
        "    assert t.allclose(win[0, 0, k], x[0, 0, k:k+3])\n"
        "\n"
        "# Case E: x_pad must not have been mutated by building win (no view side effects).\n"
        "x = t.randn(1, 2, 8, generator=rng)\n"
        "x_pad, win = cx8_padded_windows(x, K=3, P=1)\n"
        "x_pad_snapshot = x_pad.clone()\n"
        "_ = win.sum()  # just touch the view\n"
        "assert t.equal(x_pad, x_pad_snapshot), 'building win must not mutate x_pad'"
    ),
    "solution_body": (
        "def cx8_padded_windows(x, K, P):\n"
        "    B, IC, W = x.shape\n"
        "    # Atom B (conv-padding-zero): zero-pad the input.\n"
        "    x_pad = x.new_zeros(B, IC, W + 2 * P)\n"
        "    x_pad[..., P : P + W] = x\n"
        "    # Atom A (as-strided-windowing): read strides FROM THE PADDED TENSOR.\n"
        "    s_b, s_ic, s_w = x_pad.stride()\n"
        "    OW = W + 2 * P - K + 1\n"
        "    win = x_pad.as_strided(\n"
        "        size=(B, IC, OW, K),\n"
        "        stride=(s_b, s_ic, s_w, s_w),\n"
        "    )\n"
        "    return x_pad, win"
    ),
    "solution_notes": (
        "The critical move is reading `x_pad.stride()` — NOT `x.stride()`. After padding, you're "
        "looking at a new tensor with new (contiguous) strides. Hard-coding the pre-pad strides is "
        "the canonical ARENA bug: the windows would read garbage past the un-padded end of `x`. The "
        "stride-1 windowing pattern `(s_w, s_w)` for the trailing `(OW, K)` axes means 'advance by "
        "one element of the padded W axis to move between windows, and the same to walk within a "
        "window' — so adjacent windows overlap by `K - 1`."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["as-strided-windowing", "conv-padding-zero"],
    "lo": (
        "Compose zero-padding (materialize a (B, IC, W + 2P) tensor) with as_strided windowing "
        "(read the PADDED tensor's strides, build a (B, IC, OW, K) view) to set up the 1-D conv "
        "with padding as two clean stages — one allocation, one zero-copy view."
    ),
}


# ===========================================================================
# cx9 — stride-S as_strided windowing (stride > 1, the multiply-the-OW-stride pattern)
# ===========================================================================
spec_9 = {
    "atom_ids": ["as-strided-windowing", "conv-stride-downsample"],
    "subtopics": _subs(["as-strided-windowing", "conv-stride-downsample"]),
    "primary_atom": "as-strided-windowing",
    "part": "part2",
    "exercise_index": 9,
    "exercise_title": "as_strided windowing with stride > 1",
    "slug": "as-strided-windows-stride-gt-1",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Stride-1 windowing puts the source's element-stride `s_w` on BOTH the new `OW` axis and the "
        "`K` axis. For stride-`S` windowing, the change is surgical: **multiply the `OW` stride by "
        "`S`**, leave the `K` stride alone.\n\n"
        "```\n"
        "# stride-1 windowing (cx8):\n"
        "stride=(s_b, s_ic, s_w,     s_w)\n"
        "# stride-S windowing (this drill):\n"
        "stride=(s_b, s_ic, s_w * S, s_w)\n"
        "```\n\n"
        "The new `OW` stride means 'advance by `S` elements of the original W axis when moving to "
        "the next window' — i.e. SKIP `S - 1` positions between window starts. The `K` stride stays "
        "the same: within a window you still walk one element at a time.\n\n"
        "**Output length** (`conv-stride-downsample` atom): `OW = (W - K) // S + 1`. Floor division "
        "drops any partial trailing window.\n\n"
        "**Why the multiply, not a separate skip step.** as_strided does the skipping FOR you via "
        "the stride argument. You never write a loop, never call `step`. The view directly indexes "
        "into the right memory cells.\n\n"
        "**Stride-1 special case.** When `S = 1`, `s_w * 1 == s_w` and the formula collapses to the "
        "stride-1 case — same atom, same code path."
    ),
    "prompt_body": (
        "Implement `cx9_strided_windows(x, K, S)`.\n\n"
        "- `x`: float tensor of shape `(B, IC, W)`.\n"
        "- `K`: kernel width.\n"
        "- `S`: stride (>= 1).\n\n"
        "Return a `(B, IC, OW, K)` view where `OW = (W - K) // S + 1`, each window starts `S` "
        "elements after the previous one, and the view shares storage with `x` (no copy).\n\n"
        "1. **Output length** — apply the strided-conv formula `OW = (W - K) // S + 1`.\n"
        "2. **Read source strides** — `s_b, s_ic, s_w = x.stride()`.\n"
        "3. **Strided as_strided** — call `x.as_strided(size=(B, IC, OW, K), stride=(s_b, s_ic, s_w "
        "* S, s_w))`. The `s_w * S` is the load-bearing piece — that's where the skipping happens.\n\n"
        "The test:\n"
        "- Verifies `OW` matches the formula across many `(W, K, S)` configs.\n"
        "- Checks `win[..., k]` equals `x[..., k*S : k*S + K]` for several windows.\n"
        "- Confirms `data_ptr` matches (no copy).\n"
        "- Cross-checks `einsum(windows, weight)` against `F.conv1d(x, weight, stride=S)`."
    ),
    "stub_body": (
        "def cx9_strided_windows(x, K, S):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "from torch.nn import functional as F\n"
        "\n"
        "# Case A: hand-built, S=2, K=3. OW = (8-3)//2 + 1 = 3.\n"
        "x = t.arange(1.0, 9.0).reshape(1, 1, 8).contiguous()  # [1..8]\n"
        "win = cx9_strided_windows(x, K=3, S=2)\n"
        "assert tuple(win.shape) == (1, 1, 3, 3), f'shape: {tuple(win.shape)}'\n"
        "# Windows: [1,2,3], [3,4,5], [5,6,7].\n"
        "assert t.allclose(win[0, 0, 0], t.tensor([1.0, 2.0, 3.0]))\n"
        "assert t.allclose(win[0, 0, 1], t.tensor([3.0, 4.0, 5.0]))\n"
        "assert t.allclose(win[0, 0, 2], t.tensor([5.0, 6.0, 7.0]))\n"
        "\n"
        "# Case B: no-copy (must share storage with x).\n"
        "assert win.data_ptr() == x.data_ptr(), 'windows must be a view (share storage with x)'\n"
        "\n"
        "# Case C: stride == kernel size — non-overlapping tiles.\n"
        "x = t.arange(1.0, 13.0).reshape(1, 1, 12).contiguous()\n"
        "win = cx9_strided_windows(x, K=4, S=4)\n"
        "assert tuple(win.shape) == (1, 1, 3, 4)  # (12-4)//4 + 1 = 3.\n"
        "assert t.allclose(win[0, 0, 0], t.tensor([1.0, 2.0, 3.0, 4.0]))\n"
        "assert t.allclose(win[0, 0, 1], t.tensor([5.0, 6.0, 7.0, 8.0]))\n"
        "assert t.allclose(win[0, 0, 2], t.tensor([9.0, 10.0, 11.0, 12.0]))\n"
        "\n"
        "# Case D: stride-1 special case — equals plain stride-1 windowing.\n"
        "x = t.arange(1.0, 11.0).reshape(1, 1, 10).contiguous()\n"
        "win = cx9_strided_windows(x, K=3, S=1)\n"
        "assert tuple(win.shape) == (1, 1, 8, 3)  # (10-3)//1 + 1 = 8.\n"
        "for k in range(8):\n"
        "    assert t.allclose(win[0, 0, k], x[0, 0, k:k+3])\n"
        "\n"
        "# Case E: off-by-one cross-check vs F.conv1d for many configs.\n"
        "rng = t.Generator().manual_seed(9)\n"
        "for B, IC, W, K, S, OC in [\n"
        "    (1, 1, 10, 3, 2, 1),\n"
        "    (2, 3, 20, 5, 3, 4),\n"
        "    (3, 2, 32, 3, 2, 2),\n"
        "    (1, 4,  9, 3, 3, 2),  # stride == kernel\n"
        "    (1, 1, 11, 4, 2, 1),  # tests trailing-window drop\n"
        "]:\n"
        "    x = t.randn(B, IC, W, generator=rng)\n"
        "    win = cx9_strided_windows(x, K, S)\n"
        "    OW = (W - K) // S + 1\n"
        "    assert tuple(win.shape) == (B, IC, OW, K), (\n"
        "        f'OW formula wrong: predicted {OW}, got {win.shape[-2]} for W={W} K={K} S={S}'\n"
        "    )\n"
        "    # Spot-check a window.\n"
        "    for k in range(OW):\n"
        "        assert t.allclose(win[..., k, :], x[..., k*S : k*S + K]), (\n"
        "            f'window {k} content mismatch — did you forget s_w * S on the OW axis?'\n"
        "        )\n"
        "    # Cross-check vs F.conv1d.\n"
        "    weight = t.randn(OC, IC, K, generator=rng)\n"
        "    y_manual = einops.einsum(win, weight, 'b ic ow kw, oc ic kw -> b oc ow')\n"
        "    y_native = F.conv1d(x, weight, stride=S)\n"
        "    assert t.allclose(y_manual, y_native, atol=1e-4)\n"
        "\n"
        "# Case F: off-by-one trap — for W=32 K=3 S=2, OW must be 15, NOT 16.\n"
        "x = t.zeros(1, 1, 32)\n"
        "win = cx9_strided_windows(x, K=3, S=2)\n"
        "assert win.shape[-2] == 15, f'expected 15 (not 16), got {win.shape[-2]}'"
    ),
    "solution_body": (
        "def cx9_strided_windows(x, K, S):\n"
        "    B, IC, W = x.shape\n"
        "    # Atom B (conv-stride-downsample): floor division + 1 for the leading window.\n"
        "    OW = (W - K) // S + 1\n"
        "    s_b, s_ic, s_w = x.stride()\n"
        "    # Atom A (as-strided-windowing): the OW-axis stride is s_w * S — the multiply IS the\n"
        "    # skipping. The K-axis stride stays s_w (walk within a window).\n"
        "    return x.as_strided(\n"
        "        size=(B, IC, OW, K),\n"
        "        stride=(s_b, s_ic, s_w * S, s_w),\n"
        "    )"
    ),
    "solution_notes": (
        "Two-line solution; two atoms. The strided-conv formula gives `OW`; the as_strided call uses "
        "`s_w * S` on the new `OW` axis to skip `S - 1` elements between window starts. The `K` axis "
        "stride is still `s_w` — within a window you advance one element at a time. Hard-coding `S` "
        "on both axes is a common bug (would skip *within* the window too)."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["as-strided-windowing", "conv-stride-downsample"],
    "lo": (
        "Compose as_strided windowing (multiply the OW-axis stride by S, leave the K-axis stride "
        "alone) with the strided output-length formula ((W-K)//S + 1) to build the stride-S "
        "windowed view used in ARENA's strided conv1d."
    ),
}


# ===========================================================================
# cx10 — invert the conv-output-shape formula given padding (solve for same-padding)
# ===========================================================================
spec_10 = {
    "atom_ids": ["conv-output-shape", "conv-padding-zero"],
    "subtopics": _subs(["conv-output-shape", "conv-padding-zero"]),
    "primary_atom": "conv-output-shape",
    "part": "part2",
    "exercise_index": 10,
    "exercise_title": "predict + verify conv2d output shape when padding is non-zero",
    "slug": "outshape-with-padding",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "The 2-D conv output-shape formula, with stride and padding, is:\n"
        "```\n"
        "H_out = (H + 2*PH - KH) // SH + 1\n"
        "W_out = (W + 2*PW - KW) // SW + 1\n"
        "```\n\n"
        "The `2*P` term comes directly from the `conv-padding-zero` atom: padding by `P` on each "
        "side adds `2*P` to the effective input length BEFORE the kernel walks over it. The `// S + "
        "1` comes from `conv-stride-downsample` arithmetic.\n\n"
        "**Same-padding inversion.** For stride 1 and odd kernel `K`, the choice `P = (K - 1) // 2` "
        "gives `H_out == H` (the input shape is preserved). This drill exercises BOTH directions:\n"
        "- Forward: given hyperparams, predict `(B, OC, H_out, W_out)`.\n"
        "- Inverse: given `K` (odd, stride-1), compute the `P` that makes `H_out == H` (the 'same' "
        "padding).\n\n"
        "The forward direction tests the `conv-output-shape` atom with non-zero padding; the inverse "
        "tests understanding that padding cancels kernel-shrink for the same-padding regime."
    ),
    "prompt_body": (
        "Implement two functions.\n\n"
        "1. `cx10_outshape_with_pad(input_shape, out_channels, kernel_size, stride, padding)`:\n"
        "   - `input_shape`: `(B, IC, H, W)`.\n"
        "   - `kernel_size`, `stride`, `padding`: each a 2-tuple `(h_val, w_val)`.\n"
        "   - Return `(B, OC, H_out, W_out)` per the formula above.\n\n"
        "2. `cx10_same_padding(kernel_size)`:\n"
        "   - `kernel_size`: 2-tuple `(KH, KW)`. Each must be odd.\n"
        "   - Return `(PH, PW)` = `((KH - 1) // 2, (KW - 1) // 2)`.\n"
        "   - For stride 1, applying this padding makes `H_out == H` and `W_out == W` "
        "(verified by the test).\n\n"
        "The test:\n"
        "- Forward-checks `cx10_outshape_with_pad` against a real `nn.Conv2d`.\n"
        "- Inversely checks that `cx10_same_padding` paired with stride-1 yields an identity-shape "
        "conv."
    ),
    "stub_body": (
        "def cx10_outshape_with_pad(input_shape, out_channels, kernel_size, stride, padding):\n"
        "    raise NotImplementedError\n"
        "\n"
        "def cx10_same_padding(kernel_size):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "from torch import nn\n"
        "\n"
        "def _check_forward(input_shape, oc, k, s, p):\n"
        "    predicted = cx10_outshape_with_pad(input_shape, oc, k, s, p)\n"
        "    conv = nn.Conv2d(\n"
        "        in_channels=input_shape[1], out_channels=oc,\n"
        "        kernel_size=k, stride=s, padding=p,\n"
        "    )\n"
        "    x = t.zeros(*input_shape)\n"
        "    actual = tuple(conv(x).shape)\n"
        "    assert tuple(predicted) == actual, (\n"
        "        f'shape mismatch for {input_shape} oc={oc} k={k} s={s} p={p}:\\n'\n"
        "        f'  predicted {tuple(predicted)}\\n  actual    {actual}'\n"
        "    )\n"
        "\n"
        "# Case A: stride-1 same-padding (odd kernel) — output equals input on H, W.\n"
        "_check_forward((1, 3, 32, 32), 16, (3, 3), (1, 1), (1, 1))   # 3,3 same-pad\n"
        "_check_forward((1, 3, 32, 32), 16, (5, 5), (1, 1), (2, 2))   # 5,5 same-pad\n"
        "_check_forward((1, 3, 32, 32), 16, (7, 7), (1, 1), (3, 3))   # 7,7 same-pad\n"
        "\n"
        "# Case B: padding bigger than the kernel — output strictly larger than input.\n"
        "_check_forward((1, 3, 8, 8), 1, (3, 3), (1, 1), (2, 2))\n"
        "\n"
        "# Case C: non-square padding and kernel.\n"
        "_check_forward((2, 1, 28, 40), 4, (5, 3), (2, 1), (2, 1))\n"
        "\n"
        "# Case D: zero padding (no-op) — degenerates to (H-K)//S + 1.\n"
        "_check_forward((1, 3, 32, 32), 16, (3, 3), (1, 1), (0, 0))\n"
        "_check_forward((1, 3, 32, 32), 16, (3, 3), (2, 2), (0, 0))\n"
        "\n"
        "# Case E: stride > 1 with padding — the stride-2 + same-padding ResNet pattern.\n"
        "_check_forward((4, 8, 64, 64), 32, (3, 3), (2, 2), (1, 1))   # halves spatial axes cleanly\n"
        "_check_forward((1, 1, 16, 16), 1, (5, 5), (2, 2), (2, 2))\n"
        "\n"
        "# Case F: same-padding inversion — for each odd kernel, the helper produces P that makes\n"
        "# H_out == H (stride 1).\n"
        "for K in [1, 3, 5, 7, 9]:\n"
        "    P = cx10_same_padding((K, K))\n"
        "    assert P == ((K - 1) // 2, (K - 1) // 2), f'K={K}: same_padding wrong, got {P}'\n"
        "    out_shape = cx10_outshape_with_pad((1, 1, 24, 24), 1, (K, K), (1, 1), P)\n"
        "    assert tuple(out_shape) == (1, 1, 24, 24), (\n"
        "        f'same-pad K={K} P={P} should give (1,1,24,24), got {tuple(out_shape)}'\n"
        "    )\n"
        "\n"
        "# Case G: cross-check same-padding helper against nn.Conv2d for non-square odd kernels.\n"
        "for KH, KW in [(3, 5), (5, 1), (1, 7), (7, 7)]:\n"
        "    P = cx10_same_padding((KH, KW))\n"
        "    conv = nn.Conv2d(3, 8, kernel_size=(KH, KW), stride=1, padding=P)\n"
        "    x = t.zeros(1, 3, 20, 20)\n"
        "    y = conv(x)\n"
        "    assert tuple(y.shape) == (1, 8, 20, 20), (\n"
        "        f'same-pad failed: KH={KH} KW={KW} P={P} produced {tuple(y.shape)}, expected (1,8,20,20)'\n"
        "    )"
    ),
    "solution_body": (
        "def cx10_outshape_with_pad(input_shape, out_channels, kernel_size, stride, padding):\n"
        "    B, IC, H, W = input_shape\n"
        "    KH, KW = kernel_size\n"
        "    SH, SW = stride\n"
        "    PH, PW = padding\n"
        "    # Atom A (conv-output-shape) with non-zero padding term:\n"
        "    H_out = (H + 2 * PH - KH) // SH + 1\n"
        "    W_out = (W + 2 * PW - KW) // SW + 1\n"
        "    return (B, out_channels, H_out, W_out)\n"
        "\n"
        "def cx10_same_padding(kernel_size):\n"
        "    KH, KW = kernel_size\n"
        "    # Atom B (conv-padding-zero): the amount of zero-pad needed so the kernel center can\n"
        "    # reach every input cell — (K - 1) // 2 per side for stride 1, odd K.\n"
        "    return ((KH - 1) // 2, (KW - 1) // 2)"
    ),
    "solution_notes": (
        "The `2*P` term in the output-shape formula directly reflects the `conv-padding-zero` atom: "
        "each side contributes `P` extra cells the kernel can land on. Same-padding (`P = (K-1)//2`) "
        "exactly cancels the `K - 1` shrink for stride 1 — that's why ResNet's 3x3 + padding=1 keeps "
        "spatial dims constant. For stride > 1, same-padding ALONE doesn't preserve shape; it only "
        "ensures the formula divides cleanly. Forgetting the `2*` (writing `P` instead of `2*P`) is "
        "the canonical off-by-one bug here."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["conv-output-shape", "conv-padding-zero"],
    "lo": (
        "Compose the conv2d output-shape formula (with the 2P padding term) with the same-padding "
        "inversion ((K-1)//2 per side) to predict output shapes and pick padding that preserves "
        "spatial dims for odd kernels."
    ),
}


# ===========================================================================
# cx11 — conv-output-shape with stride downsample (stride > 1)
# ===========================================================================
spec_11 = {
    "atom_ids": ["conv-output-shape", "conv-stride-downsample"],
    "subtopics": _subs(["conv-output-shape", "conv-stride-downsample"]),
    "primary_atom": "conv-output-shape",
    "part": "part2",
    "exercise_index": 11,
    "exercise_title": "conv2d output shape under stride downsample",
    "slug": "outshape-with-stride",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "The 2-D conv output-shape formula expressed via the stride-downsample arithmetic:\n"
        "```\n"
        "H_out = (H + 2*PH - KH) // SH + 1\n"
        "W_out = (W + 2*PW - KW) // SW + 1\n"
        "```\n\n"
        "When `stride > 1`, the floor-division `// SH` is doing the actual downsampling. The `+ 1` "
        "captures the leading window (this is the `conv-stride-downsample` half of the formula — "
        "without `+1` you'd be off by one for *every* stride).\n\n"
        "**Two flavours of stride downsample.**\n"
        "- *Naive intuition*: 'stride 2 halves the spatial size'. WRONG without padding. For "
        "`H=32, K=3, S=2, P=0` you get `(32-3)//2 + 1 = 15`, NOT 16.\n"
        "- *Same-pad downsample* (`K=3, P=1, S=2`): `(32 + 2 - 3)//2 + 1 = 16`. The padding term "
        "cancels the kernel shrink, leaving exactly `H // S`. This is the ResNet pattern.\n\n"
        "This drill exercises both: predict the output shape AND a helper that, given input length "
        "and stride, returns the analytic 'clean halving' padding (i.e. the `P` such that `H_out == "
        "H / S` for a stride-`S`, kernel-3 conv)."
    ),
    "prompt_body": (
        "Implement two functions.\n\n"
        "1. `cx11_outshape_with_stride(input_shape, out_channels, kernel_size, stride, padding)`:\n"
        "   - Same signature as cx10 (and same formula). Return `(B, OC, H_out, W_out)`.\n\n"
        "2. `cx11_clean_halve_padding(K, S)`:\n"
        "   - Given odd kernel `K` and stride `S`, return the integer `P` such that for even input "
        "lengths `H`, `cx11_outshape_with_stride` gives `H_out == H // S`.\n"
        "   - Closed form: `P = (K - 1) // 2`. The reasoning: `(H + 2P - K) // S + 1` reduces to "
        "`H // S` when `2P = K - 1`.\n"
        "   - You may assume `K` is odd.\n\n"
        "The test cross-checks the forward function against `nn.Conv2d` and verifies the helper "
        "produces clean halving (or thirding, etc.) for several `(K, S)` combos."
    ),
    "stub_body": (
        "def cx11_outshape_with_stride(input_shape, out_channels, kernel_size, stride, padding):\n"
        "    raise NotImplementedError\n"
        "\n"
        "def cx11_clean_halve_padding(K, S):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "from torch import nn\n"
        "\n"
        "def _check_forward(input_shape, oc, k, s, p):\n"
        "    predicted = cx11_outshape_with_stride(input_shape, oc, k, s, p)\n"
        "    conv = nn.Conv2d(\n"
        "        in_channels=input_shape[1], out_channels=oc,\n"
        "        kernel_size=k, stride=s, padding=p,\n"
        "    )\n"
        "    x = t.zeros(*input_shape)\n"
        "    actual = tuple(conv(x).shape)\n"
        "    assert tuple(predicted) == actual, (\n"
        "        f'shape mismatch for {input_shape} oc={oc} k={k} s={s} p={p}:\\n'\n"
        "        f'  predicted {tuple(predicted)}\\n  actual    {actual}'\n"
        "    )\n"
        "\n"
        "# Case A: the canonical off-by-one trap. Stride 2, no pad, K=3.\n"
        "# (32-3)//2 + 1 = 15, NOT 32//2 = 16.\n"
        "predicted = cx11_outshape_with_stride((1, 1, 32, 32), 1, (3, 3), (2, 2), (0, 0))\n"
        "assert tuple(predicted) == (1, 1, 15, 15), (\n"
        "    f'expected (1,1,15,15) — off-by-one if you got 16, got {tuple(predicted)}'\n"
        ")\n"
        "\n"
        "# Case B: stride-1 no-pad (sanity).\n"
        "_check_forward((1, 1, 32, 32), 1, (3, 3), (1, 1), (0, 0))\n"
        "\n"
        "# Case C: stride > 1, no padding.\n"
        "_check_forward((1, 1, 32, 32), 1, (3, 3), (2, 2), (0, 0))\n"
        "_check_forward((1, 1, 32, 32), 1, (5, 5), (3, 3), (0, 0))\n"
        "_check_forward((2, 4, 50, 50), 8, (7, 7), (4, 4), (0, 0))\n"
        "\n"
        "# Case D: stride == kernel (non-overlapping tiles), no padding.\n"
        "_check_forward((1, 3, 24, 24), 6, (4, 4), (4, 4), (0, 0))\n"
        "\n"
        "# Case E: ResNet pattern — stride 2 + same-padding (K=3, P=1) cleanly halves even sizes.\n"
        "for H in [8, 16, 32, 64, 128]:\n"
        "    predicted = cx11_outshape_with_stride((1, 1, H, H), 1, (3, 3), (2, 2), (1, 1))\n"
        "    assert tuple(predicted) == (1, 1, H // 2, H // 2), (\n"
        "        f'clean-halve broken: H={H} got {predicted}'\n"
        "    )\n"
        "\n"
        "# Case F: non-square stride and kernel.\n"
        "_check_forward((1, 1, 17, 19), 1, (4, 2), (3, 2), (1, 0))\n"
        "\n"
        "# Case G: clean-halve helper produces the right P for each odd K.\n"
        "for K in [1, 3, 5, 7, 9]:\n"
        "    P = cx11_clean_halve_padding(K, S=2)\n"
        "    assert P == (K - 1) // 2, f'K={K}: clean_halve_padding wrong, got {P}'\n"
        "    # For each even H, applying P should yield H // 2.\n"
        "    for H in [8, 16, 32]:\n"
        "        out = cx11_outshape_with_stride((1, 1, H, H), 1, (K, K), (2, 2), (P, P))\n"
        "        assert tuple(out) == (1, 1, H // 2, H // 2), (\n"
        "            f'clean-halve K={K} P={P} H={H} produced {tuple(out)}, expected (1,1,{H//2},{H//2})'\n"
        "        )\n"
        "\n"
        "# Case H: clean-thirding (S=3) also works for odd K.\n"
        "P3 = cx11_clean_halve_padding(K=3, S=3)\n"
        "assert P3 == 1  # (3-1)//2\n"
        "# For H divisible by 3: (H + 2 - 3)//3 + 1 = (H-1)//3 + 1 = H//3 when H%3==0.\n"
        "for H in [9, 12, 15, 30]:\n"
        "    out = cx11_outshape_with_stride((1, 1, H, H), 1, (3, 3), (3, 3), (P3, P3))\n"
        "    assert tuple(out) == (1, 1, H // 3, H // 3), (\n"
        "        f'clean-thirding K=3 P=1 H={H} produced {tuple(out)}'\n"
        "    )"
    ),
    "solution_body": (
        "def cx11_outshape_with_stride(input_shape, out_channels, kernel_size, stride, padding):\n"
        "    B, IC, H, W = input_shape\n"
        "    KH, KW = kernel_size\n"
        "    SH, SW = stride\n"
        "    PH, PW = padding\n"
        "    # Atom A (conv-output-shape) — atom B (conv-stride-downsample) lives in the // SH + 1.\n"
        "    H_out = (H + 2 * PH - KH) // SH + 1\n"
        "    W_out = (W + 2 * PW - KW) // SW + 1\n"
        "    return (B, out_channels, H_out, W_out)\n"
        "\n"
        "def cx11_clean_halve_padding(K, S):\n"
        "    # Solve (H + 2P - K) // S + 1 == H // S for even H.\n"
        "    # When 2P == K - 1: (H + K - 1 - K) // S + 1 = (H - 1) // S + 1 = H // S (for H % S == 0).\n"
        "    return (K - 1) // 2"
    ),
    "solution_notes": (
        "The `+ 1` in the output-shape formula is the entire `conv-stride-downsample` atom — without "
        "it you'd be off by one for every stride. The clean-halve helper exploits the algebra: "
        "setting `2P = K - 1` collapses the formula to `(H - 1) // S + 1`, which equals `H // S` "
        "when `H` is divisible by `S`. This is why ResNet-style downsampling uses K=3, P=1, S=2: "
        "spatial dims halve cleanly with no off-by-one."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["conv-output-shape", "conv-stride-downsample"],
    "lo": (
        "Compose the conv2d output-shape formula with the stride-downsample arithmetic (// S + 1) "
        "to predict shapes under stride > 1, and invert the formula to derive the padding that "
        "produces clean H // S downsampling."
    ),
}


# ===========================================================================
# cx12 — 1-D conv from scratch: as_strided windows + einsum (the canonical ARENA recipe)
# ===========================================================================
spec_12 = {
    "atom_ids": ["as-strided-windowing", "conv-windowing-1d"],
    "subtopics": _subs(["as-strided-windowing", "conv-windowing-1d"]),
    "primary_atom": "conv-windowing-1d",
    "part": "part2",
    "exercise_index": 12,
    "exercise_title": "conv1d from scratch — as_strided windowing + einsum",
    "slug": "conv1d-via-as-strided-and-einsum",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "This is the ARENA `conv1d_minimal` recipe in one function. It decomposes a 1-D conv into "
        "TWO ops:\n"
        "1. **Windowing** (atom: `as-strided-windowing`) — build a `(B, IC, OW, K)` view onto `x` "
        "with strides `(s_b, s_ic, s_w, s_w)`. No copy.\n"
        "2. **Einsum contraction** (atom: `conv-windowing-1d` — the windowed-view-then-einsum "
        "pattern) — `einops.einsum(win, weight, 'b ic ow kw, oc ic kw -> b oc ow')`. This contracts "
        "the `IC` and `KW` axes simultaneously, dotting each window against each filter.\n\n"
        "**Why this is exactly `F.conv1d`.** Each output cell `y[b, oc, ow]` is the sum over `ic, "
        "kw` of `x[b, ic, ow + kw] * weight[oc, ic, kw]`. The windowing arranges `x` so position "
        "`(b, ic, ow, kw)` IS `x[b, ic, ow + kw]`. The einsum then evaluates the conv formula in "
        "one call.\n\n"
        "**Anatomy.**\n"
        "- `OW = W - KW + 1` (stride 1, no pad).\n"
        "- `s_b, s_ic, s_w = x.stride()`.\n"
        "- `win = x.as_strided(size=(B, IC, OW, KW), stride=(s_b, s_ic, s_w, s_w))`.\n"
        "- `y = einops.einsum(win, weight, 'b ic ow kw, oc ic kw -> b oc ow')`.\n\n"
        "**Why care.** This is the from-scratch building block ARENA uses to *implement* conv "
        "before touching `F.conv1d`. Padding and stride extensions slot in on top: pad x first "
        "(cx8); multiply the OW stride by S (cx9). This drill is the unpadded, stride-1 base case."
    ),
    "prompt_body": (
        "Implement `cx12_conv1d_from_scratch(x, weight)` — a from-scratch 1-D convolution that uses "
        "ONLY `as_strided` and `einops.einsum`. Do NOT call `F.conv1d`. Stride 1, no padding.\n\n"
        "- `x`: float tensor of shape `(B, IC, W)`.\n"
        "- `weight`: float tensor of shape `(OC, IC, KW)`.\n"
        "- Return: float tensor of shape `(B, OC, OW)` where `OW = W - KW + 1`.\n\n"
        "Two-step recipe:\n"
        "1. **Window** `x` into a `(B, IC, OW, KW)` view via `as_strided`. Read `x.stride()` for "
        "the source strides; the OW and KW axes both get stride `s_w` (stride-1 windowing pattern).\n"
        "2. **Einsum** the view against `weight`: `'b ic ow kw, oc ic kw -> b oc ow'`. This contracts "
        "the `IC` and `KW` axes in one shot.\n\n"
        "The test:\n"
        "- Cross-checks against `F.conv1d(x, weight)` to fp tolerance.\n"
        "- Verifies the intermediate windowed view shares storage with `x` (no copy in step 1).\n"
        "- Probes edge cases: `KW == W` (single window), `KW == 1` (just channel mix), batch + "
        "multi-channel."
    ),
    "stub_body": (
        "def cx12_conv1d_from_scratch(x, weight):\n"
        "    raise NotImplementedError\n"
        "\n"
        "def cx12_window_only(x, KW):\n"
        "    \"\"\"Helper exposing the windowed view so the test can verify the no-copy property.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "from torch.nn import functional as F\n"
        "\n"
        "# Case A: hand-built — single batch, single channel, kw=3.\n"
        "x = t.arange(1.0, 6.0).reshape(1, 1, 5).contiguous()  # [1,2,3,4,5]\n"
        "weight = t.tensor([[[1.0, 0.0, -1.0]]])  # (1, 1, 3)\n"
        "y = cx12_conv1d_from_scratch(x, weight)\n"
        "assert tuple(y.shape) == (1, 1, 3), f'shape: {tuple(y.shape)}'\n"
        "# Each output = x[k] - x[k+2]. y = [1-3, 2-4, 3-5] = [-2, -2, -2].\n"
        "assert t.allclose(y, t.tensor([[[-2.0, -2.0, -2.0]]])), f'y={y}'\n"
        "\n"
        "# Case B: no-copy windowing — the intermediate view must alias x.\n"
        "x = t.arange(1.0, 11.0).reshape(1, 1, 10).contiguous()\n"
        "win = cx12_window_only(x, KW=3)\n"
        "assert tuple(win.shape) == (1, 1, 8, 3)\n"
        "assert win.data_ptr() == x.data_ptr(), 'windowed view must share storage with x'\n"
        "for k in range(8):\n"
        "    assert t.allclose(win[0, 0, k], x[0, 0, k:k+3])\n"
        "\n"
        "# Case C: multi-channel, multi-filter cross-check against F.conv1d.\n"
        "rng = t.Generator().manual_seed(12)\n"
        "B, IC, W, OC, KW = 2, 3, 16, 4, 5\n"
        "x = t.randn(B, IC, W, generator=rng)\n"
        "weight = t.randn(OC, IC, KW, generator=rng)\n"
        "y_manual = cx12_conv1d_from_scratch(x, weight)\n"
        "y_native = F.conv1d(x, weight)\n"
        "assert tuple(y_manual.shape) == (B, OC, W - KW + 1)\n"
        "assert t.allclose(y_manual, y_native, atol=1e-4), (\n"
        "    f'from-scratch conv1d disagrees with F.conv1d (max diff = {(y_manual - y_native).abs().max()})'\n"
        ")\n"
        "\n"
        "# Case D: KW == W — single output cell, all of x dotted against the kernel.\n"
        "B, IC, W, OC = 2, 3, 5, 4\n"
        "x = t.randn(B, IC, W, generator=rng)\n"
        "weight = t.randn(OC, IC, W, generator=rng)\n"
        "y = cx12_conv1d_from_scratch(x, weight)\n"
        "assert tuple(y.shape) == (B, OC, 1)\n"
        "y_native = F.conv1d(x, weight)\n"
        "assert t.allclose(y, y_native, atol=1e-4)\n"
        "\n"
        "# Case E: KW == 1 — pure pointwise channel mix, identical to a 1x1 conv.\n"
        "B, IC, W, OC = 1, 5, 7, 3\n"
        "x = t.randn(B, IC, W, generator=rng)\n"
        "weight = t.randn(OC, IC, 1, generator=rng)\n"
        "y = cx12_conv1d_from_scratch(x, weight)\n"
        "assert tuple(y.shape) == (B, OC, W)\n"
        "y_native = F.conv1d(x, weight)\n"
        "assert t.allclose(y, y_native, atol=1e-4)\n"
        "\n"
        "# Case F: fuzz over many random configs.\n"
        "for B, IC, W, OC, KW in [\n"
        "    (1, 1, 8, 1, 3),\n"
        "    (3, 2, 20, 5, 4),\n"
        "    (1, 8, 12, 1, 1),\n"
        "    (2, 4, 30, 8, 7),\n"
        "]:\n"
        "    x = t.randn(B, IC, W, generator=rng)\n"
        "    weight = t.randn(OC, IC, KW, generator=rng)\n"
        "    y_manual = cx12_conv1d_from_scratch(x, weight)\n"
        "    y_native = F.conv1d(x, weight)\n"
        "    assert t.allclose(y_manual, y_native, atol=1e-4), (\n"
        "        f'mismatch for B={B} IC={IC} W={W} OC={OC} KW={KW}'\n"
        "    )"
    ),
    "solution_body": (
        "def cx12_window_only(x, KW):\n"
        "    B, IC, W = x.shape\n"
        "    OW = W - KW + 1\n"
        "    s_b, s_ic, s_w = x.stride()\n"
        "    # Atom A (as-strided-windowing): stride-1 pattern, trailing (s_w, s_w).\n"
        "    return x.as_strided(\n"
        "        size=(B, IC, OW, KW),\n"
        "        stride=(s_b, s_ic, s_w, s_w),\n"
        "    )\n"
        "\n"
        "def cx12_conv1d_from_scratch(x, weight):\n"
        "    OC, IC, KW = weight.shape\n"
        "    win = cx12_window_only(x, KW)\n"
        "    # Atom B (conv-windowing-1d): contract IC and KW in one einsum.\n"
        "    return einops.einsum(win, weight, 'b ic ow kw, oc ic kw -> b oc ow')"
    ),
    "solution_notes": (
        "Two atoms, two lines. The windowing is the load-bearing trick — once `x` is reshaped so "
        "position `(b, ic, ow, kw)` holds `x[b, ic, ow + kw]`, the conv reduces to a single einsum "
        "that contracts both IC and KW. The view shares storage with x — no allocation between "
        "input and einsum. Extending this to stride > 1 means multiplying the OW-axis stride by S "
        "(cx9); extending to non-zero padding means padding x first (cx8). This drill is the "
        "unpadded, stride-1 base case from which ARENA's full conv1d is built."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["as-strided-windowing", "conv-windowing-1d"],
    "lo": (
        "Compose as_strided windowing (build a (B, IC, OW, KW) view onto x) with the conv-windowing-1d "
        "pattern (einsum the view against weight contracting IC and KW) to implement conv1d from "
        "scratch and verify it matches F.conv1d."
    ),
}


SPECS = [spec_7, spec_8, spec_9, spec_10, spec_11, spec_12]


def main() -> None:
    for spec in SPECS:
        out = emit_composite(spec)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
