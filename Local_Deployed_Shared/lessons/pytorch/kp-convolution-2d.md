---
kc: cnn.convolution-2d
title: 2-D convolution
new_syntax: ['torch.nn.functional.conv2d']
concepts: [six-axes, functional-api]
supporting: ['cnn.convolution-1d', 'cnn.module-state', 'numpy.stack-concat-interleave', 'numpy.sorting', 'tensor.row-normalization']
previews: []
faded: [1251, 1249, 1252, 1250]
guided: []
independent: [1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261]
integrated: [1262, 1263, 1264]
---

## Concept: An image window has two spatial axes

A 2-D convolution is the 1-D one with a second spatial axis everywhere. The input is `(batch, in_channels, height, width)`, the filter bank is `(out_channels, in_channels, kh, kw)`, and the window view has six axes: batch, input channel, output row, output column, kernel row, kernel column. Output rows number `1 + (H + 2P - kh) // S` and output columns `1 + (W + 2P - kw) // S`. The view's strides are read from the (padded) input: `(bs, cs, S*hs, S*ws, hs, ws)` — moving one output row is `S` image rows, moving one kernel row is one image row, and likewise for columns. The einsum sums over input channel and both kernel axes: `"b c h w i j, o c i j -> b o h w"`.

The reason to keep six named axes rather than flattening is that every mistake in a convolution is an axis mistake — rows and columns swapped, stride applied to the kernel axis, channels summed at the wrong place — and names make each one visible. The reason to test with rectangular images and rectangular kernels is the same: a `3×3` kernel on a square image hides a row/column swap; a `2×3` kernel on a `3×5` image does not. Padding, as in 1-D, is an allocated zero tensor with `x` written into its interior on both spatial axes.

```python
import torch as t
import einops
x=t.arange(1.,10.).reshape(1,1,3,3)
windows=x.as_strided((1,1,2,2,2,2),(9,9,3,1,3,1))
w=t.ones(1,1,2,2)
y=einops.einsum(windows,w,"b c h w i j, o c i j -> b o h w")
print(y)
# Hidden checks
assert y.tolist()==[[[[12.,16.],[24.,28.]]]]
```

## Worked example

We build the window view for a rectangular image and a rectangular kernel, reading strides from the tensor. A `3×5` image with a `2×3` kernel at stride `1` gives `2` output rows and `3` output columns.

```python
import torch as t
x=t.arange(1.,16.).reshape(1,1,3,5)
bs,cs,hs,ws=x.stride()
win=x.as_strided((1,1,2,3,2,3),(bs,cs,hs,ws,hs,ws))
print(win.shape, win[0,0,1,2])
# Hidden checks
assert win.shape==(1,1,2,3,2,3) and win[0,0,1,2].tolist()==[[8.,9.,10.],[13.,14.,15.]]
```

Window `(1, 2)` is the bottom-right patch, and a kernel of ones sums it. The full output has one such sum per window position; predict the corner values before running.

```python
import einops
y=einops.einsum(win,t.ones(1,1,2,3),"b c h w i j, o c i j -> b o h w")
print(y)
# Hidden checks
assert y.tolist()==[[[[27.,33.,39.],[57.,63.,69.]]]]
```

## Faded practice

### q1251
Return x with p zeros added on every side of both spatial axes. x: float (batch,in_channels,height,width); p: zero padding on every side.

```python starter
import torch as t

def solve(x,p):
    pass
```

```python solution
import torch as t

def solve(x,p):
    b,c,h,width=x.shape
    y=t.zeros((b,c,h+2*p,width+2*p),dtype=x.dtype)
    y[:,:,p:p+h,p:p+width]=x
    return y
```

### q1249
Return the image convolution with stride s and padding p. x: float (batch,in_channels,height,width); w: float (out_channels,in_channels,kh,kw); s: stride ≥ 1; p: zero padding on every side. The kernel fits the padded input.

```python starter
import torch as t
import einops

def solve(x,w,s,p):
    pass
```

```python solution
import torch as t
import einops

def solve(x,w,s,p):
    b,c,h,width=x.shape
    o,_,kh,kw=w.shape
    xx=t.zeros((b,c,h+2*p,width+2*p),dtype=x.dtype)
    xx[:,:,p:p+h,p:p+width]=x
    bs,cs,hs,ws=xx.stride()
    oh=1+(h+2*p-kh)//s
    ow=1+(width+2*p-kw)//s
    win=xx.as_strided((b,c,oh,ow,kh,kw),(bs,cs,s*hs,s*ws,hs,ws))
    y=einops.einsum(win,w,"b c h w i j, o c i j -> b o h w")
    return y
```

## Concept: The library function performs the same convolution

`torch.nn.functional.conv2d(x, w, stride=s, padding=p)` — imported as `F.conv2d` — computes exactly the windowed einsum: same axis conventions, same output size formula, zero padding on every side, no filter flip. It takes the weight as an argument on every call, which is what makes it a *function*: it owns nothing. A `Conv2d` module is the thin wrapper that stores the weight as a trainable `Parameter` and calls the function with it; ARENA's version omits the bias because a BatchNorm that follows supplies its own offset.

The reason to write the strided version first and then switch to `F.conv2d` is verification: the library call is the oracle your hand-built windows are checked against, and once they agree you understand what the fast path does. The reason fan-in matters is initialization: each output filter reads `in_channels * kh * kw` values, and the weight scale must shrink as that count grows so outputs do not blow up — the same rule as `in_features` for a linear layer. The number of output channels sets how many filters are learned, not how many values each one reads.

```python
import torch as t
import torch.nn.functional as F
x=t.arange(1.,16.).reshape(1,1,3,5)
w=t.ones(1,1,2,3)
print(F.conv2d(x,w).shape)
# Hidden checks
assert F.conv2d(x,w).shape==(1,1,2,3)
```

## Worked example

We run the same ones-kernel over a ones-image with stride `2` and padding `1`, where the padding shows up as smaller sums at the edges. A `3×5` image padded to `5×7` with a `3×3` kernel at stride `2` gives `2` rows and `3` columns.

```python
import torch as t
import torch.nn.functional as F
x=t.ones(1,1,3,5)
w=t.ones(1,1,3,3)
y=F.conv2d(x,w,stride=2,padding=1)
print(y.shape)
# Hidden checks
assert y.shape==(1,1,2,3)
```

Corner windows overlap the padding on two sides and see four real pixels; edge windows see six. The interior of a bigger image would see nine.

```python
print(y)
# Hidden checks
assert y.tolist()==[[[[4.,6.,4.],[4.,6.,4.]]]]
```

## Faded practice

### q1252
Return the image convolution through the functional library API. x: float (batch,in_channels,height,width); w: float (out_channels,in_channels,kh,kw); s: stride ≥ 1; p: zero padding on every side. The kernel fits the padded input.

```python starter
import torch as t
import torch.nn.functional as F

def solve(x,w,s,p):
    pass
```

```python solution
import torch as t
import torch.nn.functional as F

def solve(x,w,s,p):
    return F.conv2d(x,w,stride=s,padding=p)
```

### q1250
Return the image convolution followed by ReLU. x: float (batch,in_channels,height,width); w: float (out_channels,in_channels,kh,kw); s: stride ≥ 1; p: zero padding on every side. The kernel fits the padded input.

```python starter
import torch as t
import einops

def solve(x,w,s,p):
    pass
```

```python solution
import torch as t
import einops

def solve(x,w,s,p):
    b,c,h,width=x.shape
    o,_,kh,kw=w.shape
    xx=t.zeros((b,c,h+2*p,width+2*p),dtype=x.dtype)
    xx[:,:,p:p+h,p:p+width]=x
    bs,cs,hs,ws=xx.stride()
    oh=1+(h+2*p-kh)//s
    ow=1+(width+2*p-kw)//s
    win=xx.as_strided((b,c,oh,ow,kh,kw),(bs,cs,s*hs,s*ws,hs,ws))
    y=einops.einsum(win,w,"b c h w i j, o c i j -> b o h w")
    return y.clamp(min=0)
```

## Solo practice

### q1253
Return the output height and width of the image convolution with stride s and padding p, as a tuple of two integers. x: float (batch,in_channels,height,width); w: float (out_channels,in_channels,kh,kw); s: stride ≥ 1; p: zero padding on every side. The kernel fits the padded input.

### q1254
Return average convolution activation per example and output channel, shape (batch,out_channels). x: float (batch,in_channels,height,width); w: float (out_channels,in_channels,kh,kw); s: stride ≥ 1; p: zero padding on every side. The kernel fits the padded input.

### q1255
Return image convolution with each filter adjusted to have zero total coefficient. Use stride s and padding p. x: float (batch,in_channels,height,width); w: float (out_channels,in_channels,kh,kw); s: stride ≥ 1; p: zero padding on every side. The kernel fits the padded input.

### q1256
Return each output channel’s maximum response over all examples and positions, shape (out_channels,). x: float (batch,in_channels,height,width); w: float (out_channels,in_channels,kh,kw); s: stride ≥ 1; p: zero padding on every side. The kernel fits the padded input.

### q1257
Return convolution activation at the upper-left output position, shape (batch,out_channels). Include padding exactly as in the full operation. x: float (batch,in_channels,height,width); w: float (out_channels,in_channels,kh,kw); s: stride ≥ 1; p: zero padding on every side. The kernel fits the padded input.

### q1258
Return convolution feature maps centred to zero spatial mean independently for each example and output channel. x: float (batch,in_channels,height,width); w: float (out_channels,in_channels,kh,kw); s: stride ≥ 1; p: zero padding on every side. The kernel fits the padded input.

### q1259
Return shape (batch,out_channels,2), containing each response map’s spatial minimum followed by its maximum. x: float (batch,in_channels,height,width); w: float (out_channels,in_channels,kh,kw); s: stride ≥ 1; p: zero padding on every side. The kernel fits the padded input.

### q1260
Return per-example total energy in each output channel, divided by total energy over all channels. Zero-energy examples return zeros. x: float (batch,in_channels,height,width); w: float (out_channels,in_channels,kh,kw); s: stride ≥ 1; p: zero padding on every side. The kernel fits the padded input.

### q1261
Return the response of locally mean-centred image patches to the filters. Patch means include padding and all input channels. x: float (batch,in_channels,height,width); w: float (out_channels,in_channels,kh,kw); s: stride ≥ 1; p: zero padding on every side. The kernel fits the padded input.

## Integrated practice

### q1262
Return a feature descriptor combining each channel’s mean positive activation and maximum positive activation, shape (batch,2*out_channels), means first. x: float (batch,in_channels,height,width); w: float (out_channels,in_channels,kh,kw); s: stride ≥ 1; p: zero padding on every side. The kernel fits the padded input.

### q1263
Return, for each example, the index of the output channel with the largest total activation over all positions, shape (batch,); ties choose first. Use stride s and padding p. x: float (batch,in_channels,height,width); w: float (out_channels,in_channels,kh,kw); s: stride ≥ 1; p: zero padding on every side. The kernel fits the padded input.

### q1264
Return the convolution response with every filter divided by its own Euclidean norm (each filter rescaled to unit energy), stride s and padding p. x: float (batch,in_channels,height,width); w: float (out_channels,in_channels,kh,kw); s: stride ≥ 1; p: zero padding on every side. The kernel fits the padded input.

## Misconceptions

- **Stride scales the kernel strides too.** Only the output-position strides are multiplied by `S`; kernel strides stay one image step.
- **A square test image is enough.** It hides a row/column swap; use rectangular images and kernels.
- **`F.conv2d` flips the kernel.** It does not; it is the windowed sum with the weight as given.
- **More output channels means a bigger fan-in.** Fan-in is `in_channels * kh * kw`; output channels only set how many filters exist.
