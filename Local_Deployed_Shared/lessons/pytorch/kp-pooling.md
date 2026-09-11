---
kc: cnn.pooling
title: Pooling
new_syntax: []
concepts: [global-pool, local-max]
supporting: ['cnn.convolution-2d', 'numpy.axis-reductions', 'numpy.argmin-argmax', 'tensor.row-normalization']
previews: []
faded: [1265, 1268, 1266, 1267]
guided: []
independent: [1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277]
integrated: [1278, 1279, 1280]
---

## Concept: A spatial summary keeps examples and channels apart

Global average pooling replaces each channel's whole image with a single number — its mean over height and width. On `(batch, channels, height, width)` that is `x.mean(dim=(2, 3))`, and the result is `(batch, channels)`: one row per example, one value per channel, ready for a linear classifier head. The reduction names the two spatial axes and nothing else, so examples are never averaged together and channels are never mixed.

The reason to reduce only the spatial axes is what each axis means. Averaging over the batch would blend different images; averaging over channels would blend different feature detectors. The reason the mean is the usual choice at the end of a network is that it asks "how much of this feature appeared across the image", which is robust to where it appeared; a global maximum instead asks "did this feature appear strongly anywhere". Neither has a learned weight — pooling is a fixed summary, which is why it costs nothing to train.

```python
import torch as t
x=t.tensor([[[[1.,3.],[5.,7.]],[[2.,2.],[4.,4.]]]])
print(x.mean(dim=(2,3)))
# Hidden checks
assert x.mean(dim=(2,3)).tolist()==[[4.,3.]]
```

## Worked example

We pool a batch of two one-channel images: one flat, one with a single bright pixel. The mean sees the total brightness; the maximum sees the peak. Predict both before running.

```python
import torch as t
x=t.tensor([[[[2.,2.],[2.,2.]]],[[[0.,0.],[0.,8.]]]])
print(x.mean(dim=(2,3)))
# Hidden checks
assert x.mean(dim=(2,3)).tolist()==[[2.],[2.]]
```

The two images have the same mean but very different maxima. Reducing `dim=3` then `dim=2` takes the max over columns and then rows, and `[0]` picks the values rather than the indices.

```python
print(x.max(dim=3)[0].max(dim=2)[0])
# Hidden checks
assert x.max(dim=3)[0].max(dim=2)[0].tolist()==[[2.],[8.]]
```

## Faded practice

### q1265
Return global spatial average, shape (batch,channels). x: float (batch,channels,height,width).

```python starter
import torch as t

def solve(x):
    pass
```

```python solution
import torch as t

def solve(x):
    return x.mean(dim=(2,3))
```

### q1268
Return each channel’s spatial maximum, shape (batch,channels). x: float (batch,channels,height,width).

```python starter
import torch as t

def solve(x):
    pass
```

```python solution
import torch as t

def solve(x):
    return x.max(dim=3)[0].max(dim=2)[0]
```

## Concept: Local maxima need a neutral boundary

Local max pooling keeps coarse spatial layout: each output is the largest value in one `k×k` window, with windows placed by a stride, exactly as convolution places them. The window view is the six-axis one from convolution, `(batch, channels, out_h, out_w, k, k)`, and the reduction is a maximum over the last two axes — `win.max(dim=5)[0].max(dim=4)[0]` — so channels are never combined. Output size follows the same `1 + (H + 2P - k) // S` rule.

The reason padding must be `-inf` here, not zero, is that the maximum is not neutral to zero: a window lying entirely over negative activations should return its largest negative value, and a zero pad would wrongly win. `t.full(shape, float("-inf"))` makes a boundary that can never be the maximum, and writing `x` into its interior gives the padded tensor whose strides the view reads. The reason max pooling is used between convolutions is that it halves the resolution while keeping the strongest response in each neighbourhood, so later layers see a wider context per pixel.

```python
import torch as t
x=t.tensor([-4.,-2.])
padded=t.full((4,),float("-inf"))
padded[1:3]=x
print(padded.as_strided((3,2),(1,1)).max(dim=1)[0])
# Hidden checks
assert padded.as_strided((3,2),(1,1)).max(dim=1)[0].tolist()==[-4.,-2.,-2.]
```

## Worked example

We pool a `4×4` image with `2×2` windows at stride `2`, which tiles it into four blocks. The view's output-position strides are twice the image strides; the kernel strides are the image strides themselves.

```python
import torch as t
x=t.arange(16.).reshape(1,1,4,4)
bs,cs,hs,ws=x.stride()
win=x.as_strided((1,1,2,2,2,2),(bs,cs,2*hs,2*ws,hs,ws))
print(win[0,0,0,1])
# Hidden checks
assert win[0,0,0,1].tolist()==[[2.,3.],[6.,7.]]
```

Each block's maximum is its bottom-right value, because the image increases along both axes. The result is a `2×2` map: the same layout, half the resolution.

```python
print(win.max(dim=5)[0].max(dim=4)[0])
# Hidden checks
assert win.max(dim=5)[0].max(dim=4)[0].tolist()==[[[[5.,7.],[13.,15.]]]]
```

## Faded practice

### q1266
Return local max pooling with stride s and padding p. x: float (batch,channels,height,width); k: square kernel size; s: stride; p: zero padding on every side, 0 ≤ p ≤ k//2. The kernel fits the padded image.

```python starter
import torch as t

def solve(x,k,s,p):
    pass
```

```python solution
import torch as t

def solve(x,k,s,p):
    b,c,h,w=x.shape
    xx=t.full((b,c,h+2*p,w+2*p),float("-inf"),dtype=x.dtype)
    xx[:,:,p:p+h,p:p+w]=x
    bs,cs,hs,ws=xx.stride()
    win=xx.as_strided((b,c,1+(h+2*p-k)//s,1+(w+2*p-k)//s,k,k),(bs,cs,s*hs,s*ws,hs,ws))
    y=win.max(dim=5)[0].max(dim=4)[0]
    return y
```

### q1267
Return positive parts of locally pooled values. x: float (batch,channels,height,width); k: square kernel size; s: stride; p: zero padding on every side, 0 ≤ p ≤ k//2. The kernel fits the padded image.

```python starter
import torch as t

def solve(x,k,s,p):
    pass
```

```python solution
import torch as t

def solve(x,k,s,p):
    b,c,h,w=x.shape
    xx=t.full((b,c,h+2*p,w+2*p),float("-inf"),dtype=x.dtype)
    xx[:,:,p:p+h,p:p+w]=x
    bs,cs,hs,ws=xx.stride()
    win=xx.as_strided((b,c,1+(h+2*p-k)//s,1+(w+2*p-k)//s,k,k),(bs,cs,s*hs,s*ws,hs,ws))
    y=win.max(dim=5)[0].max(dim=4)[0]
    return y.clamp(min=0)
```

## Solo practice

### q1269
Return the global average features with the mean across channels removed for each example, shape (batch,channels). x: float (batch,channels,height,width).

### q1270
Return local MIN pooling with kernel k, stride s and padding p: the smallest value in every window, where padded positions never win. x: float (batch,channels,height,width); k: square kernel size; s: stride; p: zero padding on every side, 0 ≤ p ≤ k//2. The kernel fits the padded image.

### q1271
Return the spatial range of each channel, shape (batch,channels). x: float (batch,channels,height,width).

### q1272
Return the difference between global max and global mean of each channel. x: float (batch,channels,height,width).

### q1273
Return the fraction of positive pixels in each example/channel, shape (batch,channels). x: float (batch,channels,height,width).

### q1274
Return global averages of locally max-pooled maps, shape (batch,channels). x: float (batch,channels,height,width); k: square kernel size; s: stride; p: zero padding on every side, 0 ≤ p ≤ k//2. The kernel fits the padded image.

### q1275
Return each channel’s spatial maximum location as a flattened row-major index, shape (batch,channels). Ties choose first. x: float (batch,channels,height,width).

### q1276
Return unit-length global-average feature vectors, shape (batch,channels). A zero vector stays zero. x: float (batch,channels,height,width).

### q1277
Return root-mean-square activation per example/channel, shape (batch,channels). x: float (batch,channels,height,width).

## Integrated practice

### q1278
Return concatenated global means and global maxima of locally max-pooled maps, shape (batch,2*channels), means first. x: float (batch,channels,height,width); k: square kernel size; s: stride; p: zero padding on every side, 0 ≤ p ≤ k//2. The kernel fits the padded image.

### q1279
Return the change in channelwise spatial means caused by local max pooling, shape (batch,channels): pooled mean minus original mean. x: float (batch,channels,height,width); k: square kernel size; s: stride; p: zero padding on every side, 0 ≤ p ≤ k//2. The kernel fits the padded image.

### q1280
Return a float mask marking every spatial maximum in each example/channel, preserving x shape and dtype; keep all ties. x: float (batch,channels,height,width).

## Misconceptions

- **Global pooling returns one number per image.** It returns one per channel per image, shape `(batch, channels)`.
- **Zero padding is fine for max pooling.** A zero beats every negative activation; pad with `-inf`.
- **Pooling reduces the channel axis.** It never does; channels are distinct features and stay separate.
- **`x.max(dim=(2, 3))` works like `mean`.** `max` takes one axis at a time; reduce twice and index `[0]` for the values.
