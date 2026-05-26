"""Composite drills cx13..cx18 — batch-18 (CC-cell, part2).

Six composite procedural drills exercising 2-atom pairs from the ARENA CNN
machinery (ARENA part 2 — CNNs, conv windowing + pooling).

cx13  conv-windowing-1d + einops-einsum         full 1D conv: windowed view + einsum
cx14  conv-windowing-2d + as-strided-windowing  2D windowed view for full Conv2d
cx15  conv-output-shape + conv-windowing-1d     1D output shape + window
cx16  conv-output-shape + conv-windowing-2d     2D output shape + window
cx17  maxpool-reduce    + as-strided-windowing  max pool via strided window + reduce max
cx18  avgpool-reduce    + as-strided-windowing  avg pool via strided window + reduce mean
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
# cx13 — full 1D conv: windowed view + einsum contraction
# ===========================================================================
spec_13 = {
    "atom_ids": ["conv-windowing-1d", "einops-einsum"],
    "subtopics": _subs(["conv-windowing-1d", "einops-einsum"]),
    "primary_atom": "conv-windowing-1d",
    "part": "part2",
    "exercise_index": 13,
    "exercise_title": "full 1D conv = windowed view contracted by einsum",
    "slug": "full-conv1d-as-windowed-einsum",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "ARENA's from-scratch `conv1d` decomposes into **two atoms**:\n\n"
        "1. **`conv-windowing-1d`** — turn input `x: (B, IC, W)` into a strided view "
        "`x_win: (B, IC, OW, KW)` where each `(KW,)` slice along the new `OW` axis is one "
        "kernel-sized window. Built via `x.as_strided(size=(B, IC, OW, KW), "
        "stride=(s_b, s_ic, s_w, s_w))` — **no copy**.\n"
        "2. **`einops-einsum`** — contract the window view against the kernel "
        "`weight: (OC, IC, KW)` along the shared `(ic, kw)` axes:\n"
        "   ```\n"
        "   einops.einsum(x_win, weight, 'b ic ow kw, oc ic kw -> b oc ow')\n"
        "   ```\n\n"
        "**Why this composition matters.** `F.conv1d` is a black box; this two-step "
        "decomposition is the ARENA *whitebox* form. Step 1 is pure view manipulation (free); "
        "step 2 is one fused tensor contraction. The result matches `F.conv1d(x, weight)` to "
        "fp tolerance — the proof that you understand convolution as 'sliding dot product'.\n\n"
        "**Anatomy of the einsum pattern.** Free axes (`b oc ow`) appear on the RHS. Contracted "
        "axes (`ic kw`) appear on both LHS operands but NOT on the RHS — einsum sums over them. "
        "`oc` only appears on the kernel side; `b ow` only appear on the input side. Aligning "
        "the letters across the two operands is what makes the math match `conv1d`."
    ),
    "prompt_body": (
        "Implement `cx13_conv1d_full(x, weight)` — the whitebox replacement for "
        "`F.conv1d(x, weight)` (stride=1, no padding).\n\n"
        "- `x`: float tensor of shape `(B, IC, W)`.\n"
        "- `weight`: float tensor of shape `(OC, IC, KW)`.\n"
        "- Return: tensor of shape `(B, OC, OW)` where `OW = W - KW + 1`.\n\n"
        "1. **Window** — use `x.as_strided` to build `x_win` of shape `(B, IC, OW, KW)`. "
        "Read strides from `x.stride()` — do NOT hardcode.\n"
        "2. **Einsum** — call `einops.einsum(x_win, weight, 'b ic ow kw, oc ic kw -> b oc ow')`.\n\n"
        "The test compares your output against `F.conv1d(x, weight)` on multiple shapes."
    ),
    "stub_body": (
        "def cx13_conv1d_full(x, weight):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "from torch.nn import functional as F\n"
        "rng = t.Generator().manual_seed(13)\n"
        "\n"
        "# Case A: tiny hand-checked example.\n"
        "x = t.arange(1.0, 11.0).reshape(1, 1, 10)\n"
        "w = t.tensor([[[1.0, 0.0, -1.0]]])  # (OC=1, IC=1, KW=3)\n"
        "y = cx13_conv1d_full(x, w)\n"
        "yref = F.conv1d(x, w)\n"
        "assert tuple(y.shape) == tuple(yref.shape), f'shape: {tuple(y.shape)} vs {tuple(yref.shape)}'\n"
        "assert t.allclose(y, yref, atol=1e-5), 'value mismatch on hand example'\n"
        "\n"
        "# Case B: multi-channel cross-check on several shapes.\n"
        "for B, IC, W, OC, KW in [(2,3,12,4,5),(1,1,8,1,3),(3,2,20,5,7),(2,4,16,8,1)]:\n"
        "    x2 = t.randn(B, IC, W, generator=rng)\n"
        "    w2 = t.randn(OC, IC, KW, generator=rng)\n"
        "    yref = F.conv1d(x2, w2)\n"
        "    yours = cx13_conv1d_full(x2, w2)\n"
        "    assert tuple(yours.shape) == tuple(yref.shape)\n"
        "    assert t.allclose(yours, yref, atol=1e-4), f'mismatch on B={B},IC={IC},W={W},OC={OC},KW={KW}'\n"
        "\n"
        "# Case C: edge — KW == W → single window per batch/channel.\n"
        "x3 = t.randn(2, 3, 7, generator=rng)\n"
        "w3 = t.randn(4, 3, 7, generator=rng)\n"
        "yours = cx13_conv1d_full(x3, w3)\n"
        "assert tuple(yours.shape) == (2, 4, 1)\n"
        "assert t.allclose(yours, F.conv1d(x3, w3), atol=1e-4)"
    ),
    "solution_body": (
        "def cx13_conv1d_full(x, weight):\n"
        "    B, IC, W = x.shape\n"
        "    OC, IC2, KW = weight.shape\n"
        "    assert IC == IC2, 'in_channels must match'\n"
        "    OW = W - KW + 1\n"
        "    # Atom A (conv-windowing-1d): build the (B, IC, OW, KW) strided view.\n"
        "    s_b, s_ic, s_w = x.stride()\n"
        "    x_win = x.as_strided(size=(B, IC, OW, KW), stride=(s_b, s_ic, s_w, s_w))\n"
        "    # Atom B (einops-einsum): contract over (ic, kw); keep (b, oc, ow).\n"
        "    return einops.einsum(x_win, weight, 'b ic ow kw, oc ic kw -> b oc ow')"
    ),
    "solution_notes": (
        "The two atoms ARE the implementation. There is no other code. Recognising this "
        "decomposition is what lets you generalize: add stride by multiplying `s_w * stride` on "
        "the `OW` axis, add padding by pre-padding `x` with zeros, add dilation by multiplying "
        "`s_w * dilation` on the `KW` axis. The einsum pattern stays unchanged."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["conv-windowing-1d", "einops-einsum"],
    "lo": (
        "Compose 1-D conv windowing (as_strided view) with einops.einsum (axis contraction) to "
        "implement F.conv1d from scratch."
    ),
}


# ===========================================================================
# cx14 — Conv2d window view: ARENA windowing + the generic as_strided pattern
# ===========================================================================
spec_14 = {
    "atom_ids": ["conv-windowing-2d", "as-strided-windowing"],
    "subtopics": _subs(["conv-windowing-2d", "as-strided-windowing"]),
    "primary_atom": "conv-windowing-2d",
    "part": "part2",
    "exercise_index": 14,
    "exercise_title": "build the 2D conv window view, then contract to a full Conv2d",
    "slug": "conv2d-windowed-view-and-contract",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "`conv-windowing-2d` is the 2-D specialization of `as-strided-windowing`. The generic atom "
        "says: 'to build a sliding-window view, the new axis's stride equals the source axis's stride'. "
        "Applied to the spatial `(H, W)` axes of an image tensor, this produces a `(B, IC, OH, OW, KH, KW)` "
        "tensor where each `(KH, KW)` patch is one kernel-sized window of the original image.\n\n"
        "**The stride tuple.** For input strides `(s_b, s_ic, s_h, s_w)`:\n"
        "```\n"
        "x.as_strided(\n"
        "    size=(B, IC, OH, OW, KH, KW),\n"
        "    stride=(s_b, s_ic, s_h, s_w, s_h, s_w),\n"
        ")\n"
        "```\n"
        "The trailing pair `(s_h, s_w)` appears **twice** — once for the window-position axes "
        "(`OH, OW`) and once for the within-window axes (`KH, KW`). Same numerical stride, different roles.\n\n"
        "**Contracting back to a full Conv2d.** Once you have the window view, the conv kernel "
        "`(OC, IC, KH, KW)` slots in as a tensor contraction:\n"
        "```\n"
        "einops.einsum(x_win, weight, 'b ic oh ow kh kw, oc ic kh kw -> b oc oh ow')\n"
        "```\n"
        "**Why two atoms.** The generic `as-strided-windowing` atom teaches the 'new-axis-stride-equals-"
        "source-stride' principle without committing to a 2-D shape. The CNN-specific atom is a direct "
        "application — but you should mentally derive it from the generic rule, not memorize the 6-tuple."
    ),
    "prompt_body": (
        "Implement `cx14_conv2d_full(x, weight)` — the whitebox replacement for `F.conv2d(x, weight)` "
        "(stride=1, no padding).\n\n"
        "- `x`: float tensor of shape `(B, IC, H, W)`.\n"
        "- `weight`: float tensor of shape `(OC, IC, KH, KW)`.\n"
        "- Return: tensor of shape `(B, OC, OH, OW)` where `OH = H - KH + 1`, `OW = W - KW + 1`.\n\n"
        "1. **Apply the as_strided windowing rule** — read all four strides from `x.stride()`. The new "
        "axes' strides are the *source* spatial strides; the within-window axes' strides are the SAME "
        "source spatial strides.\n"
        "2. **Build the 6-axis view** `(B, IC, OH, OW, KH, KW)` and confirm it shares storage with `x`.\n"
        "3. **Einsum-contract** against the kernel.\n\n"
        "The test cross-checks against `F.conv2d` and confirms the intermediate window view is a no-copy view."
    ),
    "stub_body": (
        "def cx14_conv2d_full(x, weight):\n"
        "    raise NotImplementedError\n"
        "\n"
        "def cx14_window_view(x, KH, KW):\n"
        "    \"\"\"Return the (B, IC, OH, OW, KH, KW) strided view of x — pre-einsum.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "from torch.nn import functional as F\n"
        "rng = t.Generator().manual_seed(14)\n"
        "\n"
        "# Case A: window-view storage sharing (no-copy property of as_strided).\n"
        "x = t.arange(1.0, 1 + 1*1*6*6).reshape(1, 1, 6, 6)\n"
        "win = cx14_window_view(x, KH=3, KW=3)\n"
        "assert tuple(win.shape) == (1, 1, 4, 4, 3, 3), f'window shape wrong: {tuple(win.shape)}'\n"
        "assert win.data_ptr() == x.data_ptr(), 'window view must share storage (no copy)'\n"
        "# Hand check: window (0,0) is x[0,0,:3,:3].\n"
        "assert t.allclose(win[0,0,0,0], x[0,0,:3,:3])\n"
        "# Window (2,1) is x[0,0,2:5,1:4].\n"
        "assert t.allclose(win[0,0,2,1], x[0,0,2:5,1:4])\n"
        "\n"
        "# Case B: full conv2d equivalence on multi-channel shapes.\n"
        "for B,IC,H,W,OC,KH,KW in [(2,3,14,16,4,5,3),(1,1,8,8,1,3,3),(3,2,12,18,6,3,5),(2,8,9,9,4,1,1)]:\n"
        "    x2 = t.randn(B, IC, H, W, generator=rng)\n"
        "    w2 = t.randn(OC, IC, KH, KW, generator=rng)\n"
        "    yref = F.conv2d(x2, w2)\n"
        "    yours = cx14_conv2d_full(x2, w2)\n"
        "    assert tuple(yours.shape) == tuple(yref.shape)\n"
        "    assert t.allclose(yours, yref, atol=1e-4), f'conv2d mismatch on {(B,IC,H,W,OC,KH,KW)}'\n"
        "\n"
        "# Case C: degenerate — KH==H, KW==W → single window per (b, ic).\n"
        "x3 = t.randn(2, 3, 5, 5, generator=rng)\n"
        "w3 = t.randn(4, 3, 5, 5, generator=rng)\n"
        "yours = cx14_conv2d_full(x3, w3)\n"
        "assert tuple(yours.shape) == (2, 4, 1, 1)\n"
        "assert t.allclose(yours, F.conv2d(x3, w3), atol=1e-4)"
    ),
    "solution_body": (
        "def cx14_window_view(x, KH, KW):\n"
        "    # Atom A (as-strided-windowing): new-axis strides = source spatial strides.\n"
        "    B, IC, H, W = x.shape\n"
        "    OH, OW = H - KH + 1, W - KW + 1\n"
        "    s_b, s_ic, s_h, s_w = x.stride()\n"
        "    return x.as_strided(\n"
        "        size=(B, IC, OH, OW, KH, KW),\n"
        "        stride=(s_b, s_ic, s_h, s_w, s_h, s_w),\n"
        "    )\n"
        "\n"
        "def cx14_conv2d_full(x, weight):\n"
        "    # Atom B (conv-windowing-2d): wire window view into the conv einsum.\n"
        "    OC, IC, KH, KW = weight.shape\n"
        "    x_win = cx14_window_view(x, KH, KW)\n"
        "    return einops.einsum(\n"
        "        x_win, weight,\n"
        "        'b ic oh ow kh kw, oc ic kh kw -> b oc oh ow',\n"
        "    )"
    ),
    "solution_notes": (
        "The `(s_h, s_w, s_h, s_w)` block is the load-bearing piece. The first pair walks WINDOWS "
        "(one input row/column per window step). The second pair walks WITHIN a window (one input "
        "row/column per kernel cell). Same source strides, different semantic axes — that's the "
        "essence of the as-strided windowing trick."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["conv-windowing-2d", "as-strided-windowing"],
    "lo": (
        "Apply the as-strided windowing rule to a 4-D image tensor to build the (B,IC,OH,OW,KH,KW) "
        "conv window view, then contract with einsum to match F.conv2d."
    ),
}


# ===========================================================================
# cx15 — 1D output shape + windowing: pre-compute OW, then build the view
# ===========================================================================
spec_15 = {
    "atom_ids": ["conv-output-shape", "conv-windowing-1d"],
    "subtopics": _subs(["conv-output-shape", "conv-windowing-1d"]),
    "primary_atom": "conv-output-shape",
    "part": "part2",
    "exercise_index": 15,
    "exercise_title": "1D conv with stride+padding: predict shape, then build window view",
    "slug": "conv1d-strided-padded-shape-then-window",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Stride-1 / no-pad conv1d lets you assume `OW = W - KW + 1`. Real ARENA conv supports "
        "**stride > 1** and **padding > 0**, so the size of the strided view depends on the output "
        "shape formula:\n\n"
        "```\n"
        "OW = (W + 2*P - KW) // S + 1\n"
        "```\n\n"
        "This is `conv-output-shape` (the 1-D specialization). Once you know `OW`, "
        "`conv-windowing-1d` builds the view — but with one twist: the OW-axis stride is now "
        "`s_w * S`, not just `s_w` (you step `S` input elements per output position).\n\n"
        "**Pad first, then window.** PyTorch zero-padding adds a contiguous halo around `x`. After "
        "padding, the padded tensor's spatial stride is its own `s_w`; use THAT stride in `as_strided`. "
        "Padding is `F.pad(x, (P, P))` for 1-D — pads only the last axis with `P` zeros on each side.\n\n"
        "**Anatomy of the composite.**\n"
        "1. Compute `OW` from the formula. (Atom A — `conv-output-shape`.)\n"
        "2. Pad `x` if `P > 0`. (Implementation detail — not an atom.)\n"
        "3. Build the strided view of shape `(B, IC, OW, KW)` with `OW`-stride `= s_w * S`. "
        "(Atom B — `conv-windowing-1d`.)\n"
        "4. Einsum-contract against the kernel."
    ),
    "prompt_body": (
        "Implement `cx15_strided_conv1d(x, weight, stride=1, padding=0)`.\n\n"
        "- `x`: float tensor `(B, IC, W)`.\n"
        "- `weight`: float tensor `(OC, IC, KW)`.\n"
        "- `stride`, `padding`: ints.\n"
        "- Return: tensor `(B, OC, OW)` matching `F.conv1d(x, weight, stride=stride, padding=padding)`.\n\n"
        "Also implement `cx15_predict_outshape(input_shape, OC, KW, stride, padding)` returning the "
        "predicted `(B, OC, OW)` tuple — the test verifies this matches the actual `F.conv1d` output.\n\n"
        "**Tip.** Use `F.pad(x, (padding, padding))` for the padding step. Compute `OW` ANALYTICALLY "
        "before windowing — do not derive it from `x.shape` after padding (you'd repeat the formula "
        "anyway)."
    ),
    "stub_body": (
        "def cx15_predict_outshape(input_shape, OC, KW, stride, padding):\n"
        "    raise NotImplementedError\n"
        "\n"
        "def cx15_strided_conv1d(x, weight, stride=1, padding=0):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "from torch.nn import functional as F\n"
        "rng = t.Generator().manual_seed(15)\n"
        "\n"
        "# Case A: output-shape predictor cross-check against F.conv1d.\n"
        "for B,IC,W,OC,KW,S,P in [(1,1,10,1,3,1,0),(2,3,16,4,5,2,1),(1,1,8,1,3,1,1),(3,2,20,5,7,3,2),(1,1,7,1,1,1,0)]:\n"
        "    pred = cx15_predict_outshape((B,IC,W), OC, KW, S, P)\n"
        "    actual = tuple(F.conv1d(t.zeros(B,IC,W), t.zeros(OC,IC,KW), stride=S, padding=P).shape)\n"
        "    assert tuple(pred) == actual, f'shape pred {pred} vs actual {actual} for {(B,IC,W,OC,KW,S,P)}'\n"
        "\n"
        "# Case B: full conv1d values on the same shapes.\n"
        "for B,IC,W,OC,KW,S,P in [(1,1,10,1,3,1,0),(2,3,16,4,5,2,1),(1,1,9,1,3,1,1),(3,2,21,5,7,3,2)]:\n"
        "    x = t.randn(B,IC,W, generator=rng)\n"
        "    w = t.randn(OC,IC,KW, generator=rng)\n"
        "    yref = F.conv1d(x, w, stride=S, padding=P)\n"
        "    yours = cx15_strided_conv1d(x, w, stride=S, padding=P)\n"
        "    assert tuple(yours.shape) == tuple(yref.shape), f'shape mismatch on {(B,IC,W,OC,KW,S,P)}'\n"
        "    assert t.allclose(yours, yref, atol=1e-4), f'value mismatch on {(B,IC,W,OC,KW,S,P)}'\n"
        "\n"
        "# Case C: stride==kernel, no pad → non-overlapping windows.\n"
        "x = t.randn(2, 3, 12, generator=rng)\n"
        "w = t.randn(4, 3, 4, generator=rng)\n"
        "yours = cx15_strided_conv1d(x, w, stride=4, padding=0)\n"
        "yref = F.conv1d(x, w, stride=4, padding=0)\n"
        "assert tuple(yours.shape) == (2, 4, 3)\n"
        "assert t.allclose(yours, yref, atol=1e-4)"
    ),
    "solution_body": (
        "def cx15_predict_outshape(input_shape, OC, KW, stride, padding):\n"
        "    # Atom A (conv-output-shape, 1-D form).\n"
        "    B, IC, W = input_shape\n"
        "    OW = (W + 2 * padding - KW) // stride + 1\n"
        "    return (B, OC, OW)\n"
        "\n"
        "def cx15_strided_conv1d(x, weight, stride=1, padding=0):\n"
        "    from torch.nn import functional as F\n"
        "    B, IC, W = x.shape\n"
        "    OC, IC2, KW = weight.shape\n"
        "    assert IC == IC2\n"
        "    # Use atom A to know OW up front.\n"
        "    _, _, OW = cx15_predict_outshape(x.shape, OC, KW, stride, padding)\n"
        "    # Pad, then window.\n"
        "    xp = F.pad(x, (padding, padding)) if padding > 0 else x\n"
        "    s_b, s_ic, s_w = xp.stride()\n"
        "    # Atom B (conv-windowing-1d) with stride-S step on the OW axis.\n"
        "    x_win = xp.as_strided(\n"
        "        size=(B, IC, OW, KW),\n"
        "        stride=(s_b, s_ic, s_w * stride, s_w),\n"
        "    )\n"
        "    return einops.einsum(x_win, weight, 'b ic ow kw, oc ic kw -> b oc ow')"
    ),
    "solution_notes": (
        "Two atoms, one composition: the shape formula tells you `OW`, the windowing trick builds the "
        "view sized to that `OW`. Decoupling 'what shape do I need' from 'how do I build it' is what "
        "lets you handle stride+padding without re-deriving from indices."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["conv-output-shape", "conv-windowing-1d"],
    "lo": (
        "Apply the 1-D conv output-shape formula with stride+padding, then build the matching strided "
        "window view to implement F.conv1d with stride and padding from scratch."
    ),
}


# ===========================================================================
# cx16 — 2D output shape + windowing: same idea, 2-D
# ===========================================================================
spec_16 = {
    "atom_ids": ["conv-output-shape", "conv-windowing-2d"],
    "subtopics": _subs(["conv-output-shape", "conv-windowing-2d"]),
    "primary_atom": "conv-output-shape",
    "part": "part2",
    "exercise_index": 16,
    "exercise_title": "2D conv with stride+padding: predict shape, then build window view",
    "slug": "conv2d-strided-padded-shape-then-window",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "The 2-D analogue of cx15. The output-shape formula now produces TWO output dims:\n\n"
        "```\n"
        "OH = (H + 2*PH - KH) // SH + 1\n"
        "OW = (W + 2*PW - KW) // SW + 1\n"
        "```\n\n"
        "Once you have `OH, OW`, the 2-D windowing trick builds the 6-axis view "
        "`(B, IC, OH, OW, KH, KW)`. With stride and padding, the OH/OW strides become "
        "`s_h * SH` and `s_w * SW` respectively; the KH/KW strides stay at `s_h, s_w`.\n\n"
        "**Padding for 2-D.** `F.pad(x, (PW, PW, PH, PH))` — last-axis padding comes first in the "
        "tuple (it's PyTorch's convention, NOT a typo). After padding, read `padded.stride()` for the "
        "fresh `(s_b, s_ic, s_h, s_w)`.\n\n"
        "**Why decouple shape from view.** ARENA's modular Conv2d class computes the output shape "
        "BEFORE building the view — because shape errors must surface as predictable assertions, not "
        "as a downstream as_strided crash with an uninformative message."
    ),
    "prompt_body": (
        "Implement `cx16_strided_conv2d(x, weight, stride=(1,1), padding=(0,0))`.\n\n"
        "- `x`: float tensor `(B, IC, H, W)`.\n"
        "- `weight`: float tensor `(OC, IC, KH, KW)`.\n"
        "- `stride`, `padding`: 2-tuples of ints.\n"
        "- Return: tensor `(B, OC, OH, OW)` matching `F.conv2d(x, weight, stride=stride, padding=padding)`.\n\n"
        "Also implement `cx16_predict_outshape(input_shape, OC, kernel_size, stride, padding)`. The "
        "test verifies the predicted shape matches the F.conv2d output shape on a battery of cases.\n\n"
        "**Tip.** For F.pad on 4-D tensors, the order is `(left, right, top, bottom)` — last axis "
        "padding first. Easy mistake: passing `(PH, PH, PW, PW)` swaps height and width."
    ),
    "stub_body": (
        "def cx16_predict_outshape(input_shape, OC, kernel_size, stride, padding):\n"
        "    raise NotImplementedError\n"
        "\n"
        "def cx16_strided_conv2d(x, weight, stride=(1,1), padding=(0,0)):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "from torch.nn import functional as F\n"
        "rng = t.Generator().manual_seed(16)\n"
        "\n"
        "# Case A: output-shape predictor cross-check.\n"
        "cases = [\n"
        "    ((1,3,32,32), 16, (3,3), (1,1), (0,0)),\n"
        "    ((1,3,32,32), 16, (3,3), (1,1), (1,1)),\n"
        "    ((4,8,64,64), 32, (3,3), (2,2), (1,1)),\n"
        "    ((2,1,28,40),  4, (5,3), (2,1), (2,1)),\n"
        "    ((1,3,24,24),  6, (4,4), (4,4), (0,0)),\n"
        "    ((1,1,17,19),  1, (4,2), (3,2), (1,0)),\n"
        "]\n"
        "for ishape, OC, k, s, p in cases:\n"
        "    pred = cx16_predict_outshape(ishape, OC, k, s, p)\n"
        "    actual = tuple(F.conv2d(t.zeros(*ishape), t.zeros(OC, ishape[1], *k), stride=s, padding=p).shape)\n"
        "    assert tuple(pred) == actual, f'shape mismatch {pred} vs {actual} for {(ishape,OC,k,s,p)}'\n"
        "\n"
        "# Case B: full conv2d values on the same shape budget.\n"
        "for ishape, OC, k, s, p in cases[:5]:\n"
        "    B, IC, H, W = ishape\n"
        "    x = t.randn(B, IC, H, W, generator=rng)\n"
        "    w = t.randn(OC, IC, *k, generator=rng)\n"
        "    yref = F.conv2d(x, w, stride=s, padding=p)\n"
        "    yours = cx16_strided_conv2d(x, w, stride=s, padding=p)\n"
        "    assert tuple(yours.shape) == tuple(yref.shape), f'shape diff on {(ishape,OC,k,s,p)}'\n"
        "    assert t.allclose(yours, yref, atol=1e-4), f'value diff on {(ishape,OC,k,s,p)}'"
    ),
    "solution_body": (
        "def cx16_predict_outshape(input_shape, OC, kernel_size, stride, padding):\n"
        "    # Atom A (conv-output-shape, 2-D form).\n"
        "    B, IC, H, W = input_shape\n"
        "    KH, KW = kernel_size\n"
        "    SH, SW = stride\n"
        "    PH, PW = padding\n"
        "    OH = (H + 2 * PH - KH) // SH + 1\n"
        "    OW = (W + 2 * PW - KW) // SW + 1\n"
        "    return (B, OC, OH, OW)\n"
        "\n"
        "def cx16_strided_conv2d(x, weight, stride=(1,1), padding=(0,0)):\n"
        "    from torch.nn import functional as F\n"
        "    B, IC, H, W = x.shape\n"
        "    OC, IC2, KH, KW = weight.shape\n"
        "    assert IC == IC2\n"
        "    SH, SW = stride\n"
        "    PH, PW = padding\n"
        "    _, _, OH, OW = cx16_predict_outshape(x.shape, OC, (KH, KW), stride, padding)\n"
        "    # Pad: F.pad on 4-D wants (left, right, top, bottom) — last-axis first.\n"
        "    xp = F.pad(x, (PW, PW, PH, PH)) if (PH > 0 or PW > 0) else x\n"
        "    s_b, s_ic, s_h, s_w = xp.stride()\n"
        "    # Atom B (conv-windowing-2d): step SH/SW on the OH/OW axes.\n"
        "    x_win = xp.as_strided(\n"
        "        size=(B, IC, OH, OW, KH, KW),\n"
        "        stride=(s_b, s_ic, s_h * SH, s_w * SW, s_h, s_w),\n"
        "    )\n"
        "    return einops.einsum(\n"
        "        x_win, weight,\n"
        "        'b ic oh ow kh kw, oc ic kh kw -> b oc oh ow',\n"
        "    )"
    ),
    "solution_notes": (
        "The OH/OW axes get multiplied strides (`s_h * SH`, `s_w * SW`) because incrementing the "
        "output position by 1 corresponds to moving `S` input cells. The KH/KW axes do NOT get "
        "multiplied — within a window, you always read every adjacent input position."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["conv-output-shape", "conv-windowing-2d"],
    "lo": (
        "Apply the 2-D conv output-shape formula with stride+padding, then build the matching strided "
        "window view to implement F.conv2d with stride and padding from scratch."
    ),
}


# ===========================================================================
# cx17 — MaxPool2d via strided window + reduce(max)
# ===========================================================================
spec_17 = {
    "atom_ids": ["maxpool-reduce", "as-strided-windowing"],
    "subtopics": _subs(["maxpool-reduce", "as-strided-windowing"]),
    "primary_atom": "maxpool-reduce",
    "part": "part2",
    "exercise_index": 17,
    "exercise_title": "MaxPool2d with stride: as_strided windowing + einops.reduce(max)",
    "slug": "maxpool2d-via-strided-window-and-reduce",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "**Non-overlapping MaxPool** (kernel==stride) is trivially `einops.reduce(x, 'b c (h p1) (w p2) "
        "-> b c h w', 'max')` — the parenthesized axis factoring does both the windowing and the "
        "reduction in one expression.\n\n"
        "**Overlapping MaxPool** (kernel ≠ stride — the ResNet stem case) needs the two atoms "
        "separately:\n\n"
        "1. **`as-strided-windowing`** — build a `(B, C, OH, OW, KH, KW)` view of the input where each "
        "`(KH, KW)` patch is one pool window. With pool stride `S`, the OH/OW strides are "
        "`s_h * S, s_w * S`; the KH/KW strides are `s_h, s_w` (within-window walks element by element).\n"
        "2. **`maxpool-reduce`** — apply `einops.reduce(..., 'b c oh ow kh kw -> b c oh ow', 'max')` "
        "to collapse the `(KH, KW)` pool patch into one scalar (its max).\n\n"
        "**Why this composition matters.** Once you can pool with strided windowing + `reduce(max)`, "
        "you can change one letter (`'max' -> 'mean'`) to get AvgPool (see cx18). The window view is "
        "shared infrastructure — it's the same trick conv uses.\n\n"
        "**Edge: kernel == stride.** Strides collapse to `(s_h * K, s_w * K, s_h, s_w)` — the "
        "non-overlapping case. The composite handles both because the formula is the same."
    ),
    "prompt_body": (
        "Implement `cx17_maxpool2d(x, kernel_size, stride=None)`.\n\n"
        "- `x`: float tensor `(B, C, H, W)`.\n"
        "- `kernel_size`: int (square kernel) — the pool window size.\n"
        "- `stride`: int or None. If `None`, defaults to `kernel_size` (non-overlapping, "
        "PyTorch's default). May be smaller than `kernel_size` (overlapping pool).\n"
        "- Return: tensor `(B, C, OH, OW)` matching `F.max_pool2d(x, kernel_size, stride=stride)` "
        "(no padding).\n\n"
        "Use `as_strided` to build the window view, then `einops.reduce` with `'max'` to pool. "
        "Compute `OH = (H - K) // S + 1`, `OW = (W - K) // S + 1`.\n\n"
        "The test fuzzes overlapping and non-overlapping cases against `F.max_pool2d`."
    ),
    "stub_body": (
        "def cx17_maxpool2d(x, kernel_size, stride=None):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "from torch.nn import functional as F\n"
        "rng = t.Generator().manual_seed(17)\n"
        "\n"
        "# Case A: hand example, non-overlapping.\n"
        "x = t.tensor([[[\n"
        "    [1.0, 2.0, 3.0, 4.0],\n"
        "    [5.0, 6.0, 7.0, 8.0],\n"
        "    [9.0, 1.0, 2.0, 3.0],\n"
        "    [4.0, 5.0, 6.0, 7.0],\n"
        "]]])\n"
        "y = cx17_maxpool2d(x, kernel_size=2)\n"
        "assert tuple(y.shape) == (1, 1, 2, 2), f'shape: {tuple(y.shape)}'\n"
        "assert t.allclose(y, t.tensor([[[[6.0, 8.0], [9.0, 7.0]]]]))\n"
        "\n"
        "# Case B: overlapping (ResNet stem case) — kernel=3, stride=2.\n"
        "for B,C,H,W,K,S in [(2,3,8,8,3,2),(1,1,16,16,3,2),(2,4,10,12,2,1),(1,2,7,7,3,2)]:\n"
        "    xr = t.randn(B, C, H, W, generator=rng)\n"
        "    yours = cx17_maxpool2d(xr, kernel_size=K, stride=S)\n"
        "    yref = F.max_pool2d(xr, kernel_size=K, stride=S)\n"
        "    assert tuple(yours.shape) == tuple(yref.shape), f'shape diff on {(B,C,H,W,K,S)}'\n"
        "    assert t.allclose(yours, yref, atol=1e-5), f'value diff on {(B,C,H,W,K,S)}'\n"
        "\n"
        "# Case C: default stride (None) → non-overlapping pool.\n"
        "xr = t.randn(2, 3, 8, 8, generator=rng)\n"
        "yours = cx17_maxpool2d(xr, kernel_size=4)\n"
        "yref = F.max_pool2d(xr, kernel_size=4)\n"
        "assert tuple(yours.shape) == (2, 3, 2, 2)\n"
        "assert t.allclose(yours, yref, atol=1e-5)\n"
        "\n"
        "# Case D: K=1 → identity (max over a 1x1 window is the value itself).\n"
        "xr = t.randn(1, 2, 4, 4, generator=rng)\n"
        "assert t.allclose(cx17_maxpool2d(xr, kernel_size=1, stride=1), xr, atol=1e-7)"
    ),
    "solution_body": (
        "def cx17_maxpool2d(x, kernel_size, stride=None):\n"
        "    K = kernel_size\n"
        "    S = K if stride is None else stride\n"
        "    B, C, H, W = x.shape\n"
        "    OH = (H - K) // S + 1\n"
        "    OW = (W - K) // S + 1\n"
        "    # Atom A (as-strided-windowing): step S on OH/OW, step 1 on KH/KW.\n"
        "    s_b, s_c, s_h, s_w = x.stride()\n"
        "    x_win = x.as_strided(\n"
        "        size=(B, C, OH, OW, K, K),\n"
        "        stride=(s_b, s_c, s_h * S, s_w * S, s_h, s_w),\n"
        "    )\n"
        "    # Atom B (maxpool-reduce): collapse the within-window axes by max.\n"
        "    return einops.reduce(x_win, 'b c oh ow kh kw -> b c oh ow', 'max')"
    ),
    "solution_notes": (
        "Notice that switching from MaxPool to AvgPool is a one-token change (`'max' -> 'mean'`). The "
        "windowing infrastructure is identical. That's the whole point of factoring pool as "
        "`window + reduce` — the reducer is a parameter."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["maxpool-reduce", "as-strided-windowing"],
    "lo": (
        "Compose as-strided windowing (build pool patch view) with einops.reduce(max) (collapse the "
        "within-window axes) to implement overlapping MaxPool2d from scratch."
    ),
}


# ===========================================================================
# cx18 — AvgPool2d via strided window + reduce(mean)
# ===========================================================================
spec_18 = {
    "atom_ids": ["avgpool-reduce", "as-strided-windowing"],
    "subtopics": _subs(["avgpool-reduce", "as-strided-windowing"]),
    "primary_atom": "avgpool-reduce",
    "part": "part2",
    "exercise_index": 18,
    "exercise_title": "AvgPool2d with stride: as_strided windowing + einops.reduce(mean)",
    "slug": "avgpool2d-via-strided-window-and-reduce",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Same skeleton as cx17 (MaxPool), with `'mean'` swapped in for `'max'`. The windowing step is "
        "literally identical — the only difference is the reducer applied to the `(KH, KW)` axes "
        "after the view is built.\n\n"
        "1. **`as-strided-windowing`** — build the `(B, C, OH, OW, KH, KW)` window view. OH/OW "
        "strides scale by pool stride `S`; KH/KW strides stay at the source spatial strides.\n"
        "2. **`avgpool-reduce`** — `einops.reduce(..., '... kh kw -> ...', 'mean')` collapses the "
        "pool patch to its mean.\n\n"
        "**Pop quiz: why does this compose so cleanly?** Pool is a *commutative-monoid reduction* "
        "over the within-window axes. Both `max` and `mean` are such reductions; so is `sum` and "
        "`prod`. Once you have the window view, ANY reducer plugs in — the view is the universal "
        "infrastructure.\n\n"
        "**ARENA AdaptiveAvgPool, briefly.** Global avg-pool (used at the end of ResNet) is the "
        "degenerate case: `K = H, S = H`, OH = OW = 1. The composite handles it without modification."
    ),
    "prompt_body": (
        "Implement `cx18_avgpool2d(x, kernel_size, stride=None)`.\n\n"
        "- `x`: float tensor `(B, C, H, W)`.\n"
        "- `kernel_size`: int (square kernel).\n"
        "- `stride`: int or None — default is `kernel_size` (non-overlapping).\n"
        "- Return: tensor `(B, C, OH, OW)` matching `F.avg_pool2d(x, kernel_size, stride=stride)` "
        "(no padding, default `count_include_pad=True` doesn't matter since we don't pad).\n\n"
        "Same recipe as cx17 — only the reducer changes. The test verifies overlapping and "
        "non-overlapping cases against `F.avg_pool2d`."
    ),
    "stub_body": (
        "def cx18_avgpool2d(x, kernel_size, stride=None):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "from torch.nn import functional as F\n"
        "rng = t.Generator().manual_seed(18)\n"
        "\n"
        "# Case A: hand example.\n"
        "x = t.tensor([[[\n"
        "    [1.0, 2.0, 3.0, 4.0],\n"
        "    [5.0, 6.0, 7.0, 8.0],\n"
        "    [9.0, 1.0, 2.0, 3.0],\n"
        "    [4.0, 5.0, 6.0, 7.0],\n"
        "]]])\n"
        "y = cx18_avgpool2d(x, kernel_size=2)\n"
        "expected = t.tensor([[[\n"
        "    [(1+2+5+6)/4, (3+4+7+8)/4],\n"
        "    [(9+1+4+5)/4, (2+3+6+7)/4],\n"
        "]]])\n"
        "assert tuple(y.shape) == (1, 1, 2, 2)\n"
        "assert t.allclose(y, expected, atol=1e-6)\n"
        "\n"
        "# Case B: overlapping and non-overlapping cross-checks.\n"
        "for B,C,H,W,K,S in [(2,3,8,8,3,2),(1,1,16,16,3,2),(2,4,10,12,2,1),(1,2,8,8,4,4),(3,2,12,12,2,3)]:\n"
        "    xr = t.randn(B, C, H, W, generator=rng)\n"
        "    yours = cx18_avgpool2d(xr, kernel_size=K, stride=S)\n"
        "    yref = F.avg_pool2d(xr, kernel_size=K, stride=S)\n"
        "    assert tuple(yours.shape) == tuple(yref.shape), f'shape on {(B,C,H,W,K,S)}'\n"
        "    assert t.allclose(yours, yref, atol=1e-5), f'value on {(B,C,H,W,K,S)}'\n"
        "\n"
        "# Case C: default stride.\n"
        "xr = t.randn(2, 3, 8, 8, generator=rng)\n"
        "yours = cx18_avgpool2d(xr, kernel_size=4)\n"
        "yref = F.avg_pool2d(xr, kernel_size=4)\n"
        "assert t.allclose(yours, yref, atol=1e-5)\n"
        "\n"
        "# Case D: constant input → constant output (mean of constants = constant).\n"
        "xc = t.full((2, 3, 6, 6), 4.2)\n"
        "yc = cx18_avgpool2d(xc, kernel_size=3, stride=3)\n"
        "assert tuple(yc.shape) == (2, 3, 2, 2)\n"
        "assert t.allclose(yc, t.full((2, 3, 2, 2), 4.2), atol=1e-6)\n"
        "\n"
        "# Case E: global pool — K==H, S==H → OH==OW==1.\n"
        "xg = t.randn(2, 4, 7, 7, generator=rng)\n"
        "yg = cx18_avgpool2d(xg, kernel_size=7, stride=7)\n"
        "assert tuple(yg.shape) == (2, 4, 1, 1)\n"
        "assert t.allclose(yg, F.avg_pool2d(xg, kernel_size=7, stride=7), atol=1e-5)"
    ),
    "solution_body": (
        "def cx18_avgpool2d(x, kernel_size, stride=None):\n"
        "    K = kernel_size\n"
        "    S = K if stride is None else stride\n"
        "    B, C, H, W = x.shape\n"
        "    OH = (H - K) // S + 1\n"
        "    OW = (W - K) // S + 1\n"
        "    # Atom A (as-strided-windowing): pool window view, no copy.\n"
        "    s_b, s_c, s_h, s_w = x.stride()\n"
        "    x_win = x.as_strided(\n"
        "        size=(B, C, OH, OW, K, K),\n"
        "        stride=(s_b, s_c, s_h * S, s_w * S, s_h, s_w),\n"
        "    )\n"
        "    # Atom B (avgpool-reduce): collapse within-window axes by mean.\n"
        "    return einops.reduce(x_win, 'b c oh ow kh kw -> b c oh ow', 'mean')"
    ),
    "solution_notes": (
        "Compare against cx17: the only difference is `'max' -> 'mean'`. The lesson is that pool "
        "operations are NOT atomic — they decompose into windowing (a view-construction trick) + "
        "reduction (a one-letter parameter). Recognising this turns AvgPool / MaxPool / SumPool / "
        "global-pool into the same op with different reducers."
    ),
    "extra_imports": [],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["avgpool-reduce", "as-strided-windowing"],
    "lo": (
        "Compose as-strided windowing with einops.reduce(mean) to implement overlapping AvgPool2d "
        "from scratch — recognise that switching the reducer is the only difference from MaxPool."
    ),
}


for spec in [spec_13, spec_14, spec_15, spec_16, spec_17, spec_18]:
    out = emit_composite(spec)
    print(f"wrote {out}")
