---
kc: cnn.convolution-1d
title: 1-D convolution from strided windows
new_syntax: []
concepts: [windows, stride-padding]
supporting: ['cnn.stride-views', 'einops.einsum', 'torch.slice-assignment', 'numpy.constructors', 'numpy.sorting', 'numpy.transpose-axes', 'tensor.row-normalization', 'numpy.stack-concat-interleave', 'python.control-flow']
previews: []
faded: [1233, 1234, 1235, 1236]
guided: []
independent: [1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245]
integrated: [1246, 1247, 1248]
---

## Concept: One filter, many windows

A convolution scores every local patch of a signal with the same small filter. For a 1-D signal of width `W` and a kernel of length `K`, there are `W - K + 1` valid windows — every run of `K` consecutive values — and output `p` is the dot product of window `p` with the filter. The procedure is: build a view whose rows are the windows, multiply each row by the filter, sum along the kernel axis. The view is `x.as_strided((W-K+1, K), (stride, stride))`: moving to the next window and moving to the next value inside a window are both one storage step, so the two strides are equal, and no window is ever copied.

With channels and a batch the same idea gains axes. Input `x` is `(batch, in_channels, width)`, the filter bank `w` is `(out_channels, in_channels, kernel)`, and the window view is `(batch, in_channels, positions, kernel)`. Each output filter reads every input channel, so the einsum sums over the channel *and* kernel axes and keeps batch, output channel and position: `"b c p k, o c k -> b o p"`. The reason PyTorch's "convolution" does not flip the filter is that a learned filter can be learned flipped; the first weight meets the first value of each window, which is what the strided view gives for free.

```python
import torch as t
x=t.tensor([1.,2.,4.,8.])
w=t.tensor([1.,-1.])
windows=x.as_strided((3,2),(1,1))
print((windows*w).sum(dim=1))
# Hidden checks
assert (windows*w).sum(dim=1).tolist()==[-1.,-2.,-4.]
```

## Worked example

We convolve one example with two input channels and a single output filter. First the window view: width `3` and kernel `2` give two positions, and the strides come from `x.stride()` rather than from assumptions.

```python
import torch as t
import einops
x=t.tensor([[[1.,2.,3.],[4.,5.,6.]]])
w=t.tensor([[[1.,0.],[0.,1.]]])
bs,cs,ws=x.stride()
windows=x.as_strided((1,2,2,2),(bs,cs,ws,ws))
print(windows[0,0], windows[0,1])
# Hidden checks
assert windows[0,0].tolist()==[[1.,2.],[2.,3.]] and windows[0,1].tolist()==[[4.,5.],[5.,6.]]
```

The filter takes the first value of channel `0` and the second value of channel `1`, so position `0` gives `1 + 5` and position `1` gives `2 + 6`. The einsum sums over `c` and `k` and keeps `b o p`.

```python
y=einops.einsum(windows,w,"b c p k, o c k -> b o p")
print(y)
# Hidden checks
assert y.tolist()==[[[6.,8.]]]
```

## Faded practice

### q1233
Return valid convolution with stride one and no padding. x: float (batch,in_channels,width); w: float (out_channels,in_channels,kernel).

```python starter
import torch as t
import einops

def solve(x,w):
    pass
```

```python solution
import torch as t
import einops

def solve(x,w):
    b,c,width=x.shape
    o,_,k=w.shape
    win=x.as_strided((b,c,width-k+1,k),(x.stride(0),x.stride(1),x.stride(2),x.stride(2)))
    y=einops.einsum(win,w,"b c p k, o c k -> b o p")
    return y
```

### q1234
Return each filter’s valid output averaged over positions. x: float (batch,in_channels,width); w: float (out_channels,in_channels,kernel).

```python starter
import torch as t
import einops

def solve(x,w):
    pass
```

```python solution
import torch as t
import einops

def solve(x,w):
    b,c,width=x.shape
    o,_,k=w.shape
    win=x.as_strided((b,c,width-k+1,k),(x.stride(0),x.stride(1),x.stride(2),x.stride(2)))
    y=einops.einsum(win,w,"b c p k, o c k -> b o p")
    return y.mean(dim=2)
```

## Concept: Stride chooses positions; padding extends the signal

Two knobs change which windows exist. A stride `S` makes consecutive windows start `S` values apart instead of one, so the view's position stride becomes `S * stride` while the kernel stride stays `stride`. Padding `P` adds `P` known values — zeros, for convolution — at each end of the signal before any window is taken, so windows can be centred on the edges. With both, the output width is `1 + (W + 2P - K) // S`: the number of window starts that fit.

The reason padding must be a real, allocated tensor is that a view cannot address storage that does not exist; so make a zero tensor of the padded shape, write `x` into its middle with a slice assignment, and take the strides from the *padded* tensor, since that is what the windows read. The reason zero is the right padding value here is that it is neutral for the weighted sum: a zero times any weight adds nothing. It is not neutral for every operation — a maximum over negative values would be wrongly beaten by a zero — which the pooling lesson handles differently.

```python
import torch as t
x=t.tensor([1.,2.,3.])
padded=t.zeros(5)
padded[1:4]=x
print(padded)
# Hidden checks
assert padded.tolist()==[0.,1.,2.,3.,0.]
```

## Worked example

We take stride-2 windows of kernel `3` over that padded signal. The formula gives `1 + (5 - 3) // 2 = 2` windows, and the position stride is `2 * 1`.

```python
import torch as t
padded=t.tensor([0.,1.,2.,3.,0.])
windows=padded.as_strided((2,3),(2*padded.stride(0),padded.stride(0)))
print(windows)
# Hidden checks
assert windows.tolist()==[[0.,1.,2.],[2.,3.,0.]]
```

Both windows include a padding zero, which contributes nothing to the sum. With a filter of all ones, each output is the sum of the real values it covers.

```python
w=t.ones(3)
print((windows*w).sum(dim=1))
# Hidden checks
assert (windows*w).sum(dim=1).tolist()==[3.,5.]
```

## Faded practice

### q1235
Return x with p zeros added at both ends of its width axis, preserving batch and channels. x: float (batch,in_channels,width); p: zeros added at each end.

```python starter
import torch as t

def solve(x,p):
    pass
```

```python solution
import torch as t

def solve(x,p):
    b,c,width=x.shape
    y=t.zeros((b,c,width+2*p),dtype=x.dtype)
    y[:,:,p:p+width]=x
    return y
```

### q1236
Return convolution with stride s and symmetric zero padding p. x: float (batch,in_channels,width); w: float (out_channels,in_channels,kernel); s: stride ≥ 1; p: zeros added at each end. The kernel fits the padded input.

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
    b,c,width=x.shape
    o,_,k=w.shape
    xx=t.zeros((b,c,width+2*p),dtype=x.dtype)
    xx[:,:,p:p+width]=x
    step=xx.stride()
    n=1+(xx.shape[2]-k)//s
    win=xx.as_strided((b,c,n,k),(step[0],step[1],s*step[2],step[2]))
    y=einops.einsum(win,w,"b c p k, o c k -> b o p")
    return y
```

## Solo practice

### q1237
Return the valid convolution (stride one, no padding) summed over output channels, shape (batch,out_width). x: float (batch,in_channels,width); w: float (out_channels,in_channels,kernel).

### q1238
Return valid convolution followed by positive-part activation. x: float (batch,in_channels,width); w: float (out_channels,in_channels,kernel).

### q1239
Return each filter’s maximum valid activation over positions. x: float (batch,in_channels,width); w: float (out_channels,in_channels,kernel).

### q1240
Return valid convolution with stride s and no padding. x: float (batch,in_channels,width); w: float (out_channels,in_channels,kernel); s: stride ≥ 1. The kernel fits the padded input.

### q1241
Return the output width of the convolution with stride s and padding p, as an integer. x: float (batch,in_channels,width); w: float (out_channels,in_channels,kernel); s: stride ≥ 1; p: zeros added at each end. The kernel fits the padded input.

### q1242
Return convolution responses to channel-centred filters: each kernel position has zero mean across input channels. x: float (batch,in_channels,width); w: float (out_channels,in_channels,kernel); s: stride ≥ 1; p: zeros added at each end. The kernel fits the padded input.

### q1243
Return each filter’s activation averaged across both batch and positions, using stride s and padding p. x: float (batch,in_channels,width); w: float (out_channels,in_channels,kernel); s: stride ≥ 1; p: zeros added at each end. The kernel fits the padded input.

### q1244
Return the convolution response summed over the batch, shape (out_channels,out_width), with stride s and padding p. x: float (batch,in_channels,width); w: float (out_channels,in_channels,kernel); s: stride ≥ 1; p: zeros added at each end. The kernel fits the padded input.

### q1245
Return filters ranked by descending mean squared activation over batch and positions. Use stride s and padding p; scores are distinct. x: float (batch,in_channels,width); w: float (out_channels,in_channels,kernel); s: stride ≥ 1; p: zeros added at each end. The kernel fits the padded input.

## Integrated practice

### q1246
Return concatenated valid convolution responses at stride one and stride two, with no padding: first all stride-one positions, then stride-two positions. x: float (batch,in_channels,width); w: float (out_channels,in_channels,kernel).

### q1247
Return nonnegative convolution features scaled to total one per example; all-zero features stay zero. x: float (batch,in_channels,width); w: float (out_channels,in_channels,kernel); s: stride ≥ 1; p: zeros added at each end. The kernel fits the padded input.

### q1248
Return responses with local input means removed before scoring each filter window. Use stride s and zero padding p; mean includes padding and averages channels and kernel positions. x: float (batch,in_channels,width); w: float (out_channels,in_channels,kernel); s: stride ≥ 1; p: zeros added at each end. The kernel fits the padded input.

## Misconceptions

- **The filter is reversed before multiplying.** Not in PyTorch; weight `0` meets the first value of each window.
- **Each output channel reads one input channel.** It reads all of them; the sum runs over channel and kernel axes together.
- **Padding is a view.** Padding allocates a new tensor; strides must be read from that tensor, not from `x`.
- **Stride `S` skips values inside a window.** It skips window *starts*; each window still holds `K` consecutive values.
