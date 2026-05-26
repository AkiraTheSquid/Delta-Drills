"""Composite drills cx1..cx6 — batch-18 part2 (AA-cell, ARENA CNN composites).

Six composite procedural drills exercising 2-atom pairs across the ARENA CNN
prereq atoms — the windowing / output-shape / einsum / channel-sum cluster
that lives behind `conv2d_minimal` and Conv2d.

cx1  as-strided-windowing + conv-output-shape   (derive output shape from input + kernel + stride/pad)
cx2  as-strided-windowing + einops-einsum       (build (B, C_out, H_out, W_out) via strided view + einsum)
cx3  conv-output-shape    + einops-einsum       (parametric output shape feeds einsum reduction)
cx4  as-strided-windowing + conv-channel-sum    (windowed view + sum over C_in axis)
cx5  conv-channel-sum     + conv-output-shape   (channel-sum reduction produces (B, C_out, H_out, W_out))
cx6  conv-channel-sum     + einops-einsum       (einsum "bchw,ochw->bohw" for channel-sum)
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
# cx1 — derive output shape from input + kernel + stride/pad, then size the
#        as_strided window view to match.
# ===========================================================================
spec_1 = {
    "atom_ids": ["as-strided-windowing", "conv-output-shape"],
    "subtopics": _subs(["as-strided-windowing", "conv-output-shape"]),
    "primary_atom": "as-strided-windowing",
    "part": "part2",
    "exercise_index": 1,
    "exercise_title": "derive Conv2d output shape, then build the matching as_strided patch view",
    "slug": "as-strided-view-sized-by-conv-output-shape",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Inside `conv2d_minimal` two atoms are joined at the hip:\n\n"
        "1. **Conv output shape** — the analytic formula\n"
        "   `H_out = (H + 2*pad - K) // stride + 1` (same for W). This is just integer arithmetic — "
        "no tensors involved. It tells you HOW MANY patch origins fit on the (padded) image.\n"
        "2. **as_strided windowing** — given those H_out / W_out counts, build a view of shape "
        "`(H_out, W_out, K, K)` over the (already-padded) image. The stride tuple uses `img.stride()` "
        "as the inner step and `img.stride() * stride` as the outer step.\n\n"
        "The composition: compute `(H_out, W_out)` analytically, THEN use those numbers as the outer "
        "two dims of the `as_strided` size tuple. The view's outer shape MUST equal the conv output "
        "spatial dims — otherwise downstream einsum / channel-sum will silently produce the wrong "
        "number of outputs.\n\n"
        "**Anatomy.**\n"
        "- `H_out = (H + 2*pad - K) // stride + 1`\n"
        "- `W_out = (W + 2*pad - K) // stride + 1`\n"
        "- `img_p = F.pad(img, (pad, pad, pad, pad))`\n"
        "- `sH, sW = img_p.stride()`  → for a contiguous padded image, `(Wp, 1)`\n"
        "- `patches = img_p.as_strided((H_out, W_out, K, K), (sH*stride, sW*stride, sH, sW))`"
    ),
    "prompt_body": (
        "Implement `cx1_patches_from_shape(img, K, stride, pad)`.\n\n"
        "- `img`: float tensor of shape `(H, W)` — a single-channel image, contiguous.\n"
        "- `K`: kernel side length (int).\n"
        "- `stride`: int stride along both spatial dims.\n"
        "- `pad`: zero-padding width applied symmetrically on both spatial dims (use "
        "`F.pad(img, (pad, pad, pad, pad))`).\n\n"
        "Return a tuple `((H_out, W_out), patches)`:\n"
        "- `(H_out, W_out)`: int output spatial dims computed via the conv-output-shape formula.\n"
        "- `patches`: float tensor view of shape `(H_out, W_out, K, K)` containing every `KxK` "
        "stride-`stride` window of the PADDED image. Must be a view (storage-shared with "
        "`F.pad(img, ...)` — store it before strided-viewing it).\n\n"
        "1. **conv-output-shape atom** — compute `H_out` and `W_out` from `H, W, K, stride, pad`.\n"
        "2. **Pad the image** — `img_p = F.pad(img, (pad, pad, pad, pad))`.\n"
        "3. **as-strided-windowing atom** — call `img_p.as_strided(size, stride_tuple)` ONCE.\n\n"
        "The test fuzzes random `(H, W, K, stride, pad)` and cross-checks against a nested-loop "
        "reference, and verifies the analytic shape matches the view's shape."
    ),
    "stub_body": (
        "def cx1_patches_from_shape(img, K, stride, pad):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: stride=1, pad=0 — classic im2col.\n"
        "img = t.arange(20, dtype=t.float32).reshape(4, 5)\n"
        "(H_out, W_out), patches = cx1_patches_from_shape(img, K=2, stride=1, pad=0)\n"
        "assert (H_out, W_out) == (3, 4), f'expected (3,4), got {(H_out, W_out)}'\n"
        "assert tuple(patches.shape) == (3, 4, 2, 2)\n"
        "assert t.allclose(patches[0, 0], t.tensor([[0.0, 1.0], [5.0, 6.0]]))\n"
        "assert t.allclose(patches[2, 3], t.tensor([[13.0, 14.0], [18.0, 19.0]]))\n"
        "\n"
        "# Case B: stride=2, pad=0 — every other origin.\n"
        "img = t.arange(36, dtype=t.float32).reshape(6, 6)\n"
        "(H_out, W_out), patches = cx1_patches_from_shape(img, K=3, stride=2, pad=0)\n"
        "# (6 + 0 - 3) // 2 + 1 = 2.\n"
        "assert (H_out, W_out) == (2, 2), f'expected (2,2), got {(H_out, W_out)}'\n"
        "assert tuple(patches.shape) == (2, 2, 3, 3)\n"
        "# Patch [0,0] starts at img[0,0]; patch [0,1] starts at img[0,2] (stride=2).\n"
        "assert t.allclose(patches[0, 0], img[0:3, 0:3])\n"
        "assert t.allclose(patches[0, 1], img[0:3, 2:5])\n"
        "assert t.allclose(patches[1, 0], img[2:5, 0:3])\n"
        "\n"
        "# Case C: stride=1, pad=1 — output spatial dims must equal H (same-conv style).\n"
        "img = t.arange(16, dtype=t.float32).reshape(4, 4)\n"
        "(H_out, W_out), patches = cx1_patches_from_shape(img, K=3, stride=1, pad=1)\n"
        "# (4 + 2 - 3) // 1 + 1 = 4.\n"
        "assert (H_out, W_out) == (4, 4), f'expected (4,4) for same-conv style, got {(H_out, W_out)}'\n"
        "assert tuple(patches.shape) == (4, 4, 3, 3)\n"
        "# Top-left patch is the padded corner: zeros in the first row/col.\n"
        "tl = patches[0, 0]\n"
        "assert tl[0, 0].item() == 0.0 and tl[0, 1].item() == 0.0 and tl[0, 2].item() == 0.0\n"
        "assert tl[1, 0].item() == 0.0 and tl[2, 0].item() == 0.0\n"
        "# Center of patch [0,0] reads img[0, 0].\n"
        "assert tl[1, 1].item() == img[0, 0].item()\n"
        "\n"
        "# Case D: fuzz vs nested-loop reference.\n"
        "rng = t.Generator().manual_seed(101)\n"
        "for (H, W, K, S, P) in [(7, 9, 3, 1, 0), (8, 8, 3, 2, 1), (10, 6, 2, 2, 0), (5, 5, 5, 1, 2)]:\n"
        "    im = t.randn(H, W, generator=rng)\n"
        "    (Ho, Wo), patches = cx1_patches_from_shape(im, K=K, stride=S, pad=P)\n"
        "    # Analytic shape check.\n"
        "    assert Ho == (H + 2 * P - K) // S + 1\n"
        "    assert Wo == (W + 2 * P - K) // S + 1\n"
        "    assert tuple(patches.shape) == (Ho, Wo, K, K)\n"
        "    # Nested-loop reference on the padded image.\n"
        "    import torch.nn.functional as F\n"
        "    im_p = F.pad(im, (P, P, P, P))\n"
        "    for i in range(Ho):\n"
        "        for j in range(Wo):\n"
        "            ref = im_p[i*S : i*S + K, j*S : j*S + K]\n"
        "            assert t.allclose(patches[i, j], ref), f'patch[{i},{j}] mismatch'"
    ),
    "solution_body": (
        "def cx1_patches_from_shape(img, K, stride, pad):\n"
        "    H, W = img.shape\n"
        "    # Atom A (conv-output-shape): the closed-form output spatial dims.\n"
        "    H_out = (H + 2 * pad - K) // stride + 1\n"
        "    W_out = (W + 2 * pad - K) // stride + 1\n"
        "    # Pad first — the windowing view will read from the PADDED storage.\n"
        "    img_p = F.pad(img, (pad, pad, pad, pad))\n"
        "    sH, sW = img_p.stride()  # (W + 2*pad, 1) for a contiguous padded image\n"
        "    # Atom B (as-strided-windowing): outer dims walk patch origins with `stride`;\n"
        "    # inner dims walk inside one patch with the contiguous-image strides.\n"
        "    patches = img_p.as_strided(\n"
        "        size=(H_out, W_out, K, K),\n"
        "        stride=(sH * stride, sW * stride, sH, sW),\n"
        "    )\n"
        "    return (H_out, W_out), patches"
    ),
    "solution_notes": (
        "The two atoms have to AGREE on `(H_out, W_out)`. If your analytic formula is off-by-one and "
        "your stride tuple isn't, `as_strided` will silently read past the storage and produce "
        "garbage patches with no exception. Always compute the shape first, then size the view to "
        "match — never the other way around."
    ),
    "extra_imports": ["import torch.nn as nn", "import torch.nn.functional as F"],
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["as-strided-windowing", "conv-output-shape"],
    "lo": (
        "Compose the conv-output-shape formula (analytic H_out / W_out from input, kernel, stride, "
        "pad) with as_strided windowing (size the (H_out, W_out, K, K) view to match) to extract "
        "every conv-input patch from the padded image in one view."
    ),
}


# ===========================================================================
# cx2 — full 2D conv-forward via as_strided patches + einops.einsum
# ===========================================================================
spec_2 = {
    "atom_ids": ["as-strided-windowing", "einops-einsum"],
    "subtopics": _subs(["as-strided-windowing", "einops-einsum"]),
    "primary_atom": "as-strided-windowing",
    "part": "part2",
    "exercise_index": 2,
    "exercise_title": "Conv2d forward via as_strided patches + einops.einsum",
    "slug": "as-strided-then-einsum-conv-forward",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "ARENA's `conv2d_minimal` is essentially two atoms:\n\n"
        "1. **as-strided-windowing** — build a 6-D view of all input patches with shape "
        "`(B, C_in, H_out, W_out, K, K)`. The outer four dims index a patch; the inner two index "
        "inside one patch.\n"
        "2. **einops-einsum** — contract that view against the weight tensor "
        "`(C_out, C_in, K, K)` along the `C_in, K, K` axes. The result is "
        "`(B, C_out, H_out, W_out)` — the conv output.\n\n"
        "The einsum string spells out the joint shape:\n"
        "`einsum(patches, weight, 'b c h w kh kw, o c kh kw -> b o h w')`\n\n"
        "Two label notes:\n"
        "- `c` and `kh/kw` appear on BOTH inputs → contracted (summed over).\n"
        "- `b`, `h`, `w`, `o` appear on ONE side → preserved as output dims.\n\n"
        "The composition: this drill does the FULL forward — strided view, then one einsum call. "
        "Reference checks against `t.nn.functional.conv2d`."
    ),
    "prompt_body": (
        "Implement `cx2_conv2d_via_einsum(x, w)`.\n\n"
        "- `x`: float tensor of shape `(B, C_in, H, W)`. Assume contiguous.\n"
        "- `w`: float tensor of shape `(C_out, C_in, K, K)`. Assume square kernels, stride=1, pad=0.\n\n"
        "Return a float tensor of shape `(B, C_out, H_out, W_out)` matching "
        "`F.conv2d(x, w, bias=None, stride=1, padding=0)` up to fp32 tolerance.\n\n"
        "1. **conv-output-shape** — compute `H_out = H - K + 1`, `W_out = W - K + 1`.\n"
        "2. **as-strided-windowing** — build a 6-D view of shape `(B, C_in, H_out, W_out, K, K)`. "
        "The stride tuple uses `x.stride()` for the `(B, C_in, K, K)` inner motions and the spatial "
        "strides again for the `(H_out, W_out)` outer motions.\n"
        "3. **einops-einsum** — `einops.einsum(patches, w, 'b c h w kh kw, o c kh kw -> b o h w')`.\n\n"
        "The test cross-checks against `F.conv2d` on randomized inputs."
    ),
    "stub_body": (
        "def cx2_conv2d_via_einsum(x, w):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: minimal shape — B=1, C_in=1, C_out=1, identity-ish kernel.\n"
        "x = t.arange(16, dtype=t.float32).reshape(1, 1, 4, 4)\n"
        "w = t.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])  # (1,1,2,2) — picks top-left of each patch.\n"
        "out = cx2_conv2d_via_einsum(x, w)\n"
        "assert tuple(out.shape) == (1, 1, 3, 3)\n"
        "ref = F.conv2d(x, w)\n"
        "assert t.allclose(out, ref, atol=1e-5), f'mismatch:\\nout=\\n{out}\\nref=\\n{ref}'\n"
        "\n"
        "# Case B: multi-channel, multi-output.\n"
        "rng = t.Generator().manual_seed(202)\n"
        "x = t.randn(2, 3, 6, 7, generator=rng)\n"
        "w = t.randn(4, 3, 3, 3, generator=rng)\n"
        "out = cx2_conv2d_via_einsum(x, w)\n"
        "ref = F.conv2d(x, w)\n"
        "assert tuple(out.shape) == tuple(ref.shape), f'shape mismatch: {tuple(out.shape)} vs {tuple(ref.shape)}'\n"
        "assert t.allclose(out, ref, atol=1e-4), f'value mismatch, max abs diff = {(out - ref).abs().max()}'\n"
        "\n"
        "# Case C: K=1 — degenerate case (point conv).\n"
        "x = t.randn(2, 5, 4, 4, generator=rng)\n"
        "w = t.randn(3, 5, 1, 1, generator=rng)\n"
        "out = cx2_conv2d_via_einsum(x, w)\n"
        "ref = F.conv2d(x, w)\n"
        "assert tuple(out.shape) == (2, 3, 4, 4)\n"
        "assert t.allclose(out, ref, atol=1e-4)\n"
        "\n"
        "# Case D: non-square spatial input + larger kernel.\n"
        "x = t.randn(1, 2, 9, 5, generator=rng)\n"
        "w = t.randn(2, 2, 4, 3, generator=rng)\n"
        "out = cx2_conv2d_via_einsum(x, w)\n"
        "ref = F.conv2d(x, w)\n"
        "assert tuple(out.shape) == (1, 2, 6, 3)\n"
        "assert t.allclose(out, ref, atol=1e-4)"
    ),
    "solution_body": (
        "def cx2_conv2d_via_einsum(x, w):\n"
        "    B, C_in, H, W = x.shape\n"
        "    C_out, _, KH, KW = w.shape\n"
        "    # conv-output-shape (stride=1, pad=0).\n"
        "    H_out = H - KH + 1\n"
        "    W_out = W - KW + 1\n"
        "    # Atom A (as-strided-windowing): build (B, C_in, H_out, W_out, KH, KW) view.\n"
        "    sB, sC, sH, sW = x.stride()\n"
        "    patches = x.as_strided(\n"
        "        size=(B, C_in, H_out, W_out, KH, KW),\n"
        "        stride=(sB, sC, sH, sW, sH, sW),\n"
        "    )\n"
        "    # Atom B (einops-einsum): contract c, kh, kw between patches and weight.\n"
        "    return einops.einsum(patches, w, 'b c h w kh kw, o c kh kw -> b o h w')"
    ),
    "solution_notes": (
        "The stride tuple `(sB, sC, sH, sW, sH, sW)` is the load-bearing piece — the spatial strides "
        "appear TWICE because the same 2-D pattern walks both the patch origin (outer (H_out, W_out)) "
        "and the inside of one patch (inner (KH, KW)). Once the view exists, einsum does the entire "
        "conv arithmetic in a single contraction — no nested loops, no im2col copy."
    ),
    "extra_imports": ["import torch.nn as nn", "import torch.nn.functional as F"],
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["as-strided-windowing", "einops-einsum"],
    "lo": (
        "Compose as_strided windowing (build a (B, C_in, H_out, W_out, KH, KW) view of all conv "
        "input patches) with einops.einsum (contract c/kh/kw against the weight) to implement a full "
        "Conv2d forward pass matching F.conv2d."
    ),
}


# ===========================================================================
# cx3 — parametric output shape from conv formula feeds einsum reduction
# ===========================================================================
spec_3 = {
    "atom_ids": ["conv-output-shape", "einops-einsum"],
    "subtopics": _subs(["conv-output-shape", "einops-einsum"]),
    "primary_atom": "conv-output-shape",
    "part": "part2",
    "exercise_index": 3,
    "exercise_title": "parametric Conv2d output shape feeds einsum-over-flat-patches reduction",
    "slug": "conv-output-shape-into-einsum-reduction",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Two roles in a conv forward:\n"
        "1. **conv-output-shape** — the analytic `(H_out, W_out)` formula. Pure integer math.\n"
        "2. **einops-einsum** — the actual contraction that produces the conv output values.\n\n"
        "The bridge: einsum's OUTPUT shape is derived from the labels on the right side of the "
        "string (`-> b o h w`), but the actual `H` and `W` dims of those output axes are determined "
        "by the input patch tensor's shape — which the caller built using `H_out` and `W_out` from "
        "the conv-output-shape formula.\n\n"
        "This drill exercises BOTH atoms by:\n"
        "- (a) asking you to compute `(H_out, W_out)` for a parametric `(stride, pad)` setting, then\n"
        "- (b) running the einsum reduction over a PRE-COMPUTED `(B, C_in, H_out, W_out, K, K)` "
        "  patch tensor that the test passes you. Your job is JUST the einsum step — but you must "
        "  return `(H_out, W_out)` alongside the output for the test to verify.\n\n"
        "This isolates the conv-output-shape arithmetic and the einsum contraction without dragging "
        "in `as_strided` mechanics, so the test can fuzz a wide range of `(stride, pad)` settings."
    ),
    "prompt_body": (
        "Implement `cx3_einsum_with_shape(patches, w, H, W, K, stride, pad)`.\n\n"
        "- `patches`: float tensor of shape `(B, C_in, H_out, W_out, K, K)` — the test gives you the "
        "pre-built strided view; you do NOT need to construct it.\n"
        "- `w`: float tensor of shape `(C_out, C_in, K, K)`.\n"
        "- `H, W, K, stride, pad`: input dims and conv params (ints).\n\n"
        "Return a tuple `((H_out, W_out), out)`:\n"
        "- `(H_out, W_out)`: int output spatial dims from the conv-output-shape formula. The test "
        "asserts these match the analytic formula AND match `patches.shape[2:4]`.\n"
        "- `out`: float tensor of shape `(B, C_out, H_out, W_out)` from one einsum call against "
        "`patches` and `w`.\n\n"
        "1. **conv-output-shape atom** — `H_out = (H + 2*pad - K) // stride + 1` (same for W).\n"
        "2. **einops-einsum atom** — `einops.einsum(patches, w, 'b c h w kh kw, o c kh kw -> b o h w')`."
    ),
    "stub_body": (
        "def cx3_einsum_with_shape(patches, w, H, W, K, stride, pad):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "def _build_patches(x, K, stride, pad):\n"
        "    '''Helper used by the test to build (B, C_in, H_out, W_out, K, K) input patches.'''\n"
        "    B, C_in, H, W = x.shape\n"
        "    H_out = (H + 2 * pad - K) // stride + 1\n"
        "    W_out = (W + 2 * pad - K) // stride + 1\n"
        "    x_p = F.pad(x, (pad, pad, pad, pad))\n"
        "    sB, sC, sH, sW = x_p.stride()\n"
        "    return x_p.as_strided(\n"
        "        size=(B, C_in, H_out, W_out, K, K),\n"
        "        stride=(sB, sC, sH * stride, sW * stride, sH, sW),\n"
        "    )\n"
        "\n"
        "# Case A: stride=1, pad=0 — vanilla case.\n"
        "rng = t.Generator().manual_seed(301)\n"
        "x = t.randn(2, 3, 6, 7, generator=rng)\n"
        "w = t.randn(4, 3, 3, 3, generator=rng)\n"
        "patches = _build_patches(x, K=3, stride=1, pad=0)\n"
        "(H_out, W_out), out = cx3_einsum_with_shape(patches, w, H=6, W=7, K=3, stride=1, pad=0)\n"
        "assert (H_out, W_out) == (4, 5), f'expected (4,5), got {(H_out, W_out)}'\n"
        "assert tuple(out.shape) == (2, 4, 4, 5)\n"
        "ref = F.conv2d(x, w, stride=1, padding=0)\n"
        "assert t.allclose(out, ref, atol=1e-4)\n"
        "\n"
        "# Case B: stride=2, pad=0.\n"
        "x = t.randn(1, 2, 8, 8, generator=rng)\n"
        "w = t.randn(3, 2, 3, 3, generator=rng)\n"
        "patches = _build_patches(x, K=3, stride=2, pad=0)\n"
        "(H_out, W_out), out = cx3_einsum_with_shape(patches, w, H=8, W=8, K=3, stride=2, pad=0)\n"
        "assert (H_out, W_out) == (3, 3)\n"
        "ref = F.conv2d(x, w, stride=2, padding=0)\n"
        "assert t.allclose(out, ref, atol=1e-4)\n"
        "\n"
        "# Case C: stride=1, pad=1 — same-conv style.\n"
        "x = t.randn(1, 2, 5, 5, generator=rng)\n"
        "w = t.randn(2, 2, 3, 3, generator=rng)\n"
        "patches = _build_patches(x, K=3, stride=1, pad=1)\n"
        "(H_out, W_out), out = cx3_einsum_with_shape(patches, w, H=5, W=5, K=3, stride=1, pad=1)\n"
        "assert (H_out, W_out) == (5, 5)\n"
        "ref = F.conv2d(x, w, stride=1, padding=1)\n"
        "assert t.allclose(out, ref, atol=1e-4)\n"
        "\n"
        "# Case D: fuzz shape consistency — `(H_out, W_out)` must match `patches.shape[2:4]`.\n"
        "for (H, W, K, S, P) in [(7, 9, 3, 1, 0), (8, 8, 3, 2, 1), (10, 6, 2, 2, 0), (5, 5, 5, 1, 2)]:\n"
        "    x = t.randn(1, 1, H, W, generator=rng)\n"
        "    w = t.randn(1, 1, K, K, generator=rng)\n"
        "    patches = _build_patches(x, K=K, stride=S, pad=P)\n"
        "    (Ho, Wo), out = cx3_einsum_with_shape(patches, w, H=H, W=W, K=K, stride=S, pad=P)\n"
        "    assert Ho == (H + 2 * P - K) // S + 1, f'analytic mismatch: {Ho} vs formula'\n"
        "    assert (Ho, Wo) == tuple(patches.shape[2:4])\n"
        "    ref = F.conv2d(x, w, stride=S, padding=P)\n"
        "    assert t.allclose(out, ref, atol=1e-4)"
    ),
    "solution_body": (
        "def cx3_einsum_with_shape(patches, w, H, W, K, stride, pad):\n"
        "    # Atom A (conv-output-shape): closed-form analytic shape.\n"
        "    H_out = (H + 2 * pad - K) // stride + 1\n"
        "    W_out = (W + 2 * pad - K) // stride + 1\n"
        "    # Atom B (einops-einsum): contract c, kh, kw between patches and weight.\n"
        "    out = einops.einsum(patches, w, 'b c h w kh kw, o c kh kw -> b o h w')\n"
        "    return (H_out, W_out), out"
    ),
    "solution_notes": (
        "Returning `(H_out, W_out)` alongside `out` is the test's hook to verify both atoms "
        "independently: the analytic value AND its agreement with `patches.shape[2:4]`. In real "
        "ARENA code these two numbers ALWAYS line up — if they don't, you've miscomputed the conv "
        "shape and your einsum is silently producing garbage."
    ),
    "extra_imports": ["import torch.nn as nn", "import torch.nn.functional as F"],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["conv-output-shape", "einops-einsum"],
    "lo": (
        "Compose the conv-output-shape formula (analytic H_out / W_out from H, W, K, stride, pad) "
        "with einops.einsum (contract over c/kh/kw) to produce a (B, C_out, H_out, W_out) conv "
        "output whose spatial dims agree with the analytic formula."
    ),
}


# ===========================================================================
# cx4 — windowed view + explicit sum over C_in axis (channel-sum reduction)
# ===========================================================================
spec_4 = {
    "atom_ids": ["as-strided-windowing", "conv-channel-sum"],
    "subtopics": _subs(["as-strided-windowing", "conv-channel-sum"]),
    "primary_atom": "as-strided-windowing",
    "part": "part2",
    "exercise_index": 4,
    "exercise_title": "conv via as_strided windowing + explicit sum-reduce over C_in",
    "slug": "as-strided-then-sum-over-cin",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Most Conv2d explanations focus on the spatial axes, but the C_in axis matters too:\n\n"
        "- **conv-channel-sum** — every output channel `o` is `sum over c_in of (input_channel_c * "
        "filter_c)`. That \"sum over c_in\" is what makes Conv2d a TENSOR operation rather than a "
        "stack of per-channel 2-D convs.\n"
        "- **as-strided-windowing** — gives you a `(B, C_in, H_out, W_out, K, K)` patch view. The "
        "C_in axis is already there — you just need to reduce it.\n\n"
        "The composition: with the patch view in hand, the conv arithmetic for ONE output channel "
        "is `(patches * w[o]).sum(dim=(1, 4, 5))` — multiply broadcasts the C_in / K / K axes, then "
        "an explicit `.sum(dim=...)` collapses them. The C_in axis is the one we care about here — "
        "it's the canonical channel-mixing reduction.\n\n"
        "Why an explicit sum and not einsum? **Auditability**. When you're debugging \"why are my "
        "filters producing all-zeros after init\", a per-axis `.sum` is easier to break into "
        "intermediate prints than an einsum string. ARENA's `conv2d_minimal` uses einsum; ARENA's "
        "debugging exercises sometimes use the explicit form."
    ),
    "prompt_body": (
        "Implement `cx4_conv2d_via_sum(x, w)`.\n\n"
        "- `x`: float tensor of shape `(B, C_in, H, W)`. Stride=1, pad=0.\n"
        "- `w`: float tensor of shape `(C_out, C_in, K, K)`.\n\n"
        "Return a float tensor of shape `(B, C_out, H_out, W_out)` matching `F.conv2d(x, w)`.\n\n"
        "1. **as-strided-windowing** — build a 6-D view of shape `(B, C_in, H_out, W_out, K, K)`.\n"
        "2. **conv-channel-sum** — for each output channel:\n"
        "   - broadcast `patches` against `w[o]` (insert a leading 1 in `w` and a singleton "
        "C_out-style dim — easiest: process one `o` at a time in Python, then stack), OR\n"
        "   - do it in one go: `(patches.unsqueeze(1) * w.view(1, C_out, C_in, 1, 1, K, K)).sum"
        "(dim=(2, -2, -1))` — multiply elementwise against the broadcast weight, then explicitly "
        "`.sum` over `C_in`, `KH`, `KW` axes.\n\n"
        "The test asserts the result agrees with `F.conv2d` AND probes the per-channel sum "
        "decomposition (by checking that zeroing one C_in axis of `w` zeros the corresponding "
        "contribution)."
    ),
    "stub_body": (
        "def cx4_conv2d_via_sum(x, w):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: minimal — agrees with F.conv2d on randomized small inputs.\n"
        "rng = t.Generator().manual_seed(401)\n"
        "x = t.randn(2, 3, 5, 6, generator=rng)\n"
        "w = t.randn(4, 3, 3, 3, generator=rng)\n"
        "out = cx4_conv2d_via_sum(x, w)\n"
        "ref = F.conv2d(x, w)\n"
        "assert tuple(out.shape) == (2, 4, 3, 4), f'shape {tuple(out.shape)}'\n"
        "assert t.allclose(out, ref, atol=1e-4), f'max diff = {(out - ref).abs().max()}'\n"
        "\n"
        "# Case B: zero one input channel — the per-channel-sum atom predicts a specific drop.\n"
        "x = t.randn(1, 3, 4, 4, generator=rng)\n"
        "w = t.randn(2, 3, 3, 3, generator=rng)\n"
        "out_full = cx4_conv2d_via_sum(x, w)\n"
        "# Now zero out C_in=0 contribution by zeroing w's [:, 0, :, :].\n"
        "w_zero0 = w.clone()\n"
        "w_zero0[:, 0, :, :] = 0\n"
        "out_no0 = cx4_conv2d_via_sum(x, w_zero0)\n"
        "# The difference should equal the contribution of C_in=0 alone.\n"
        "# This validates that the per-C_in summand really is being summed (channel-sum atom).\n"
        "w_only0 = t.zeros_like(w)\n"
        "w_only0[:, 0, :, :] = w[:, 0, :, :]\n"
        "out_only0 = cx4_conv2d_via_sum(x, w_only0)\n"
        "assert t.allclose(out_full, out_no0 + out_only0, atol=1e-4), (\n"
        "    'channel-sum decomposition broken — out_full must equal sum over per-C_in contributions'\n"
        ")\n"
        "\n"
        "# Case C: K=1 + multi-channel — channel-sum is the entire operation here (no spatial mix).\n"
        "x = t.randn(2, 5, 3, 3, generator=rng)\n"
        "w = t.randn(3, 5, 1, 1, generator=rng)\n"
        "out = cx4_conv2d_via_sum(x, w)\n"
        "# At K=1 conv2d == matmul along C_in, so verify via einsum reference.\n"
        "ref = t.einsum('bchw,oc->bohw', x, w.squeeze(-1).squeeze(-1))\n"
        "assert tuple(out.shape) == (2, 3, 3, 3)\n"
        "assert t.allclose(out, ref, atol=1e-4)\n"
        "\n"
        "# Case D: larger kernel, non-square spatial.\n"
        "x = t.randn(1, 2, 9, 5, generator=rng)\n"
        "w = t.randn(2, 2, 4, 3, generator=rng)\n"
        "out = cx4_conv2d_via_sum(x, w)\n"
        "ref = F.conv2d(x, w)\n"
        "assert tuple(out.shape) == (1, 2, 6, 3)\n"
        "assert t.allclose(out, ref, atol=1e-4)"
    ),
    "solution_body": (
        "def cx4_conv2d_via_sum(x, w):\n"
        "    B, C_in, H, W = x.shape\n"
        "    C_out, _, KH, KW = w.shape\n"
        "    H_out = H - KH + 1\n"
        "    W_out = W - KW + 1\n"
        "    # Atom A (as-strided-windowing): (B, C_in, H_out, W_out, KH, KW) patch view.\n"
        "    sB, sC, sH, sW = x.stride()\n"
        "    patches = x.as_strided(\n"
        "        size=(B, C_in, H_out, W_out, KH, KW),\n"
        "        stride=(sB, sC, sH, sW, sH, sW),\n"
        "    )\n"
        "    # Atom B (conv-channel-sum): explicit reduce over C_in, KH, KW.\n"
        "    # Broadcast shapes:\n"
        "    #   patches: (B, 1,     C_in, H_out, W_out, KH, KW)\n"
        "    #   w:       (1, C_out, C_in, 1,     1,     KH, KW)\n"
        "    p = patches.unsqueeze(1)\n"
        "    wb = w.view(1, C_out, C_in, 1, 1, KH, KW)\n"
        "    prod = p * wb  # (B, C_out, C_in, H_out, W_out, KH, KW)\n"
        "    return prod.sum(dim=(2, -2, -1))  # sum over C_in, KH, KW -> (B, C_out, H_out, W_out)"
    ),
    "solution_notes": (
        "The `.sum(dim=(2, -2, -1))` is the channel-sum atom made explicit: `dim=2` is C_in (after "
        "the broadcast unsqueeze), `dim=-2, -1` are KH, KW. This is the explicit-reduce form of the "
        "same operation einsum would do — slower but more debuggable. The Case-B sanity (zero one "
        "C_in row of w, expect that channel's contribution to disappear) is the canonical test of "
        "the channel-sum atom."
    ),
    "extra_imports": ["import torch.nn as nn", "import torch.nn.functional as F"],
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["as-strided-windowing", "conv-channel-sum"],
    "lo": (
        "Compose as_strided windowing (build the (B, C_in, H_out, W_out, KH, KW) patch view) with "
        "the conv-channel-sum reduction (explicit .sum over the C_in axis after broadcasting against "
        "the weight) to produce the Conv2d forward output as a sum of per-input-channel "
        "contributions."
    ),
}


# ===========================================================================
# cx5 — channel-sum reduction MUST produce (B, C_out, H_out, W_out) shape
# ===========================================================================
spec_5 = {
    "atom_ids": ["conv-channel-sum", "conv-output-shape"],
    "subtopics": _subs(["conv-channel-sum", "conv-output-shape"]),
    "primary_atom": "conv-channel-sum",
    "part": "part2",
    "exercise_index": 5,
    "exercise_title": "channel-sum reduction produces shape predicted by conv-output-shape",
    "slug": "channel-sum-yields-conv-output-shape",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "The Conv2d output shape `(B, C_out, H_out, W_out)` is jointly determined by TWO atoms:\n\n"
        "1. **conv-output-shape** — the spatial dims `H_out`, `W_out` come from the analytic formula "
        "`(H + 2*pad - K) // stride + 1`.\n"
        "2. **conv-channel-sum** — the channel dim `C_out` comes from the WEIGHT, not the input — "
        "and the C_in axis disappears under summation. The output has `C_out` channels because we "
        "sum-reduce per-output-channel.\n\n"
        "If either atom is wrong the shape is wrong:\n"
        "- Forget to reduce over C_in → output has an extra C_in axis you never wanted.\n"
        "- Get the conv-output-shape formula off by one → spatial dims don't match downstream.\n\n"
        "The composition: this drill makes you write a `channel_sum_conv(x, w, stride, pad)` whose "
        "JOB is to produce the exactly-correct `(B, C_out, H_out, W_out)` tensor — with stride+pad "
        "passed parametrically, AND with the C_in reduction explicit. The test asserts both pieces "
        "independently: shape (conv-output-shape) and value (conv-channel-sum)."
    ),
    "prompt_body": (
        "Implement `cx5_channel_sum_conv(x, w, stride=1, pad=0)`.\n\n"
        "- `x`: float tensor of shape `(B, C_in, H, W)`.\n"
        "- `w`: float tensor of shape `(C_out, C_in, K, K)`.\n"
        "- `stride`, `pad`: ints. Match `F.conv2d` semantics.\n\n"
        "Return a float tensor of shape `(B, C_out, H_out, W_out)` matching `F.conv2d(x, w, "
        "stride=stride, padding=pad)`.\n\n"
        "1. **conv-output-shape atom** — compute `H_out`, `W_out` via the formula.\n"
        "2. Pad the input via `F.pad`.\n"
        "3. Build a strided patch view of shape `(B, C_in, H_out, W_out, K, K)`.\n"
        "4. **conv-channel-sum atom** — broadcast against `w` and `.sum` over the C_in (and "
        "kernel) axes to collapse C_in. The resulting C_out axis comes from the weight tensor.\n\n"
        "Critical invariant the test checks: `out.shape[1] == w.shape[0]` (C_out from weight, NOT "
        "from input), `out.shape[2:] == (H_out, W_out)` (from conv-output-shape formula)."
    ),
    "stub_body": (
        "def cx5_channel_sum_conv(x, w, stride=1, pad=0):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: stride=1, pad=0 — vanilla. Shape and value both checked.\n"
        "rng = t.Generator().manual_seed(501)\n"
        "x = t.randn(2, 3, 6, 7, generator=rng)\n"
        "w = t.randn(4, 3, 3, 3, generator=rng)\n"
        "out = cx5_channel_sum_conv(x, w, stride=1, pad=0)\n"
        "# Shape predicted by conv-output-shape formula:\n"
        "assert out.shape[0] == 2 and out.shape[1] == 4, f'B / C_out wrong: {tuple(out.shape)}'\n"
        "assert tuple(out.shape[2:]) == (4, 5), f'(H_out,W_out) wrong: {tuple(out.shape[2:])}'\n"
        "# Value matches F.conv2d:\n"
        "ref = F.conv2d(x, w)\n"
        "assert t.allclose(out, ref, atol=1e-4)\n"
        "\n"
        "# Case B: stride=2 — shape changes per conv-output-shape, value still matches.\n"
        "x = t.randn(1, 2, 8, 8, generator=rng)\n"
        "w = t.randn(3, 2, 3, 3, generator=rng)\n"
        "out = cx5_channel_sum_conv(x, w, stride=2, pad=0)\n"
        "# (8 - 3) // 2 + 1 = 3.\n"
        "assert tuple(out.shape) == (1, 3, 3, 3)\n"
        "ref = F.conv2d(x, w, stride=2)\n"
        "assert t.allclose(out, ref, atol=1e-4)\n"
        "\n"
        "# Case C: pad=1 — same-conv-style spatial shape, value matches.\n"
        "x = t.randn(1, 2, 5, 5, generator=rng)\n"
        "w = t.randn(2, 2, 3, 3, generator=rng)\n"
        "out = cx5_channel_sum_conv(x, w, stride=1, pad=1)\n"
        "# (5 + 2 - 3) // 1 + 1 = 5.\n"
        "assert tuple(out.shape) == (1, 2, 5, 5)\n"
        "ref = F.conv2d(x, w, stride=1, padding=1)\n"
        "assert t.allclose(out, ref, atol=1e-4)\n"
        "\n"
        "# Case D: changing C_in must NOT change the output's C_out. The reduction MUST collapse C_in.\n"
        "C_in_a, C_in_b = 3, 7\n"
        "x_a = t.randn(1, C_in_a, 5, 5, generator=rng)\n"
        "x_b = t.randn(1, C_in_b, 5, 5, generator=rng)\n"
        "w_a = t.randn(4, C_in_a, 3, 3, generator=rng)\n"
        "w_b = t.randn(4, C_in_b, 3, 3, generator=rng)\n"
        "out_a = cx5_channel_sum_conv(x_a, w_a, stride=1, pad=0)\n"
        "out_b = cx5_channel_sum_conv(x_b, w_b, stride=1, pad=0)\n"
        "assert out_a.shape == out_b.shape, (\n"
        "    'output shape must NOT depend on C_in — the channel-sum atom is what collapses it'\n"
        ")\n"
        "# Both must equal F.conv2d outputs separately.\n"
        "assert t.allclose(out_a, F.conv2d(x_a, w_a), atol=1e-4)\n"
        "assert t.allclose(out_b, F.conv2d(x_b, w_b), atol=1e-4)"
    ),
    "solution_body": (
        "def cx5_channel_sum_conv(x, w, stride=1, pad=0):\n"
        "    B, C_in, H, W = x.shape\n"
        "    C_out, _, KH, KW = w.shape\n"
        "    # Atom A (conv-output-shape).\n"
        "    H_out = (H + 2 * pad - KH) // stride + 1\n"
        "    W_out = (W + 2 * pad - KW) // stride + 1\n"
        "    # Pad + strided view.\n"
        "    x_p = F.pad(x, (pad, pad, pad, pad))\n"
        "    sB, sC, sH, sW = x_p.stride()\n"
        "    patches = x_p.as_strided(\n"
        "        size=(B, C_in, H_out, W_out, KH, KW),\n"
        "        stride=(sB, sC, sH * stride, sW * stride, sH, sW),\n"
        "    )\n"
        "    # Atom B (conv-channel-sum): elementwise broadcast then .sum over (C_in, KH, KW).\n"
        "    p = patches.unsqueeze(1)                              # (B, 1, C_in, H_out, W_out, KH, KW)\n"
        "    wb = w.view(1, C_out, C_in, 1, 1, KH, KW)\n"
        "    return (p * wb).sum(dim=(2, -2, -1))                  # (B, C_out, H_out, W_out)"
    ),
    "solution_notes": (
        "The shape invariant is the key check: `out.shape[1]` comes from `w.shape[0]` (C_out, the "
        "weight's output channel count), NOT from `x.shape[1]` (C_in). That's what `conv-channel-sum` "
        "is FOR — collapsing C_in. The Case-D paired check (different C_in, same output shape) is "
        "exactly this invariant. `conv-output-shape` is what makes the spatial dims correct."
    ),
    "extra_imports": ["import torch.nn as nn", "import torch.nn.functional as F"],
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["conv-channel-sum", "conv-output-shape"],
    "lo": (
        "Compose the conv-channel-sum reduction (explicit .sum over C_in collapsing the input "
        "channel axis) with the conv-output-shape formula (analytic H_out / W_out from H, W, K, "
        "stride, pad) to produce a Conv2d output tensor of exactly (B, C_out, H_out, W_out) shape."
    ),
}


# ===========================================================================
# cx6 — channel-sum via einsum string "bchw,ochw->bohw"
# ===========================================================================
spec_6 = {
    "atom_ids": ["conv-channel-sum", "einops-einsum"],
    "subtopics": _subs(["conv-channel-sum", "einops-einsum"]),
    "primary_atom": "conv-channel-sum",
    "part": "part2",
    "exercise_index": 6,
    "exercise_title": "1x1 conv as channel-sum via einsum 'bchw,ochw->bohw'",
    "slug": "channel-sum-as-einsum-bchw-ochw-bohw",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "A `1x1` convolution is the pure channel-sum atom with no spatial windowing:\n"
        "  `out[b, o, h, w] = sum_c x[b, c, h, w] * w[o, c, h, w]`  (spatially aligned).\n\n"
        "More commonly the 1x1 conv is written with a `(C_out, C_in, 1, 1)` weight — but if we "
        "imagine the spatially-broadcast version where `w` has shape `(C_out, C_in, H, W)`, the "
        "operation becomes a clean einsum:\n\n"
        "  `einops.einsum(x, w, 'b c h w, o c h w -> b o h w')`\n\n"
        "Two atoms:\n"
        "- **conv-channel-sum** — the `c` label appears on BOTH inputs but not on the output → "
        "  einsum contracts over c, summing the C_in axis.\n"
        "- **einops-einsum** — `h, w` appear on both inputs AND the output → preserved (no "
        "  reduction along spatial axes; this is the channel-only flavor of a conv).\n\n"
        "The composition: write the einsum, and verify it matches a hand-rolled channel-sum "
        "reduction over the same shapes."
    ),
    "prompt_body": (
        "Implement `cx6_channel_sum_einsum(x, w)`.\n\n"
        "- `x`: float tensor of shape `(B, C_in, H, W)`.\n"
        "- `w`: float tensor of shape `(C_out, C_in, H, W)` — note: SAME spatial dims as `x`. This "
        "is the spatially-broadcast form of a per-pixel channel mix.\n\n"
        "Return a float tensor of shape `(B, C_out, H, W)` such that\n"
        "  `out[b, o, h, w] = sum_c x[b, c, h, w] * w[o, c, h, w]`.\n\n"
        "Use ONE `einops.einsum` call. The string MUST contract over `c` and preserve `b, o, h, w`.\n\n"
        "1. **einops-einsum atom** — string of the form `'b c h w, o c h w -> b o h w'`.\n"
        "2. **conv-channel-sum atom** — the only contracted axis is `c` (C_in), which is what makes "
        "this the channel-sum reduction.\n\n"
        "The test cross-checks against the explicit `.sum(dim=2)` form and against a 1x1-conv "
        "reference."
    ),
    "stub_body": (
        "def cx6_channel_sum_einsum(x, w):\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: hand-built shape, fixed values, cross-check against explicit .sum.\n"
        "rng = t.Generator().manual_seed(601)\n"
        "x = t.randn(2, 3, 4, 5, generator=rng)\n"
        "w = t.randn(4, 3, 4, 5, generator=rng)\n"
        "out = cx6_channel_sum_einsum(x, w)\n"
        "assert tuple(out.shape) == (2, 4, 4, 5)\n"
        "# Reference: explicit broadcast + sum over C_in.\n"
        "ref_sum = (x.unsqueeze(1) * w.unsqueeze(0)).sum(dim=2)  # (B, C_out, H, W)\n"
        "assert t.allclose(out, ref_sum, atol=1e-4), f'max diff {(out - ref_sum).abs().max()}'\n"
        "\n"
        "# Case B: scalar channels — degenerate but verifies the einsum string is correct.\n"
        "x = t.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # (1, 1, 2, 2)\n"
        "w = t.tensor([[[[10.0, 20.0], [30.0, 40.0]]]])  # (1, 1, 2, 2)\n"
        "out = cx6_channel_sum_einsum(x, w)\n"
        "# sum_c is over a single channel, so out == x * w pointwise.\n"
        "expected = t.tensor([[[[10.0, 40.0], [90.0, 160.0]]]])\n"
        "assert t.allclose(out, expected, atol=1e-6), f'\\nout={out}\\nexpected={expected}'\n"
        "\n"
        "# Case C: 1x1 conv reference — broadcast w spatially from a (C_out, C_in, 1, 1) weight.\n"
        "x = t.randn(2, 3, 4, 4, generator=rng)\n"
        "w_1x1 = t.randn(5, 3, 1, 1, generator=rng)\n"
        "# Build the spatially-broadcast (C_out, C_in, H, W) weight.\n"
        "w_full = w_1x1.expand(5, 3, 4, 4).contiguous()\n"
        "out = cx6_channel_sum_einsum(x, w_full)\n"
        "ref_1x1 = F.conv2d(x, w_1x1)  # (2, 5, 4, 4)\n"
        "assert tuple(out.shape) == (2, 5, 4, 4)\n"
        "assert t.allclose(out, ref_1x1, atol=1e-4), 'channel-sum einsum must match 1x1 conv'\n"
        "\n"
        "# Case D: zero one C_in row of w, expect that channel's contribution to vanish.\n"
        "x = t.randn(1, 3, 3, 3, generator=rng)\n"
        "w = t.randn(2, 3, 3, 3, generator=rng)\n"
        "out_full = cx6_channel_sum_einsum(x, w)\n"
        "w_z = w.clone(); w_z[:, 0, :, :] = 0  # kill C_in=0.\n"
        "out_no0 = cx6_channel_sum_einsum(x, w_z)\n"
        "w_o = t.zeros_like(w); w_o[:, 0, :, :] = w[:, 0, :, :]\n"
        "out_only0 = cx6_channel_sum_einsum(x, w_o)\n"
        "assert t.allclose(out_full, out_no0 + out_only0, atol=1e-4), (\n"
        "    'channel-sum decomposition broken — einsum must sum over c'\n"
        ")"
    ),
    "solution_body": (
        "def cx6_channel_sum_einsum(x, w):\n"
        "    # Atoms: einops-einsum string with `c` shared on both inputs but absent from output\n"
        "    # = einsum contracts over c = the conv-channel-sum reduction.\n"
        "    return einops.einsum(x, w, 'b c h w, o c h w -> b o h w')"
    ),
    "solution_notes": (
        "Reading the einsum: `c` is shared but missing from the right → contraction over C_in. "
        "Every other label (`b, o, h, w`) is preserved unchanged. This is the cleanest possible "
        "channel-sum reduction — no spatial windowing, no kernel. ARENA's full `conv2d_minimal` "
        "differs only by adding `kh kw` labels to both inputs (so they contract too)."
    ),
    "extra_imports": ["import torch.nn as nn", "import torch.nn.functional as F"],
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["conv-channel-sum", "einops-einsum"],
    "lo": (
        "Compose einops.einsum (with a label shared on both inputs and absent from the output) with "
        "the conv-channel-sum reduction (contraction along C_in) to implement the per-pixel "
        "channel-mix operation that is the heart of a 1x1 conv."
    ),
}


SPECS = [spec_1, spec_2, spec_3, spec_4, spec_5, spec_6]


def main() -> None:
    for spec in SPECS:
        out = emit_composite(spec)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
