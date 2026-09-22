---
kc: cnn.residual-block
title: The residual block
new_syntax: ['torch.nn.Conv2d', 'torch.nn.BatchNorm2d', 'torch.nn.Identity']
concepts: [left-branch, shortcut-and-sum]
supporting: ['cnn.sequential', 'cnn.convolution-2d', 'cnn.batch-normalization', 'python.control-flow']
previews: []
faded: [1389, 1390, 1391, 1392]
guided: []
independent: [1393, 1394, 1395, 1396, 1397, 1398]
integrated: [1399, 1400, 1401]
---

## Concept: The left branch is conv, norm, ReLU, conv, norm

The convolution and batch-norm lessons built both operations by hand; the library packages them as layers. `t.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=s, padding=1, bias=False)` owns a weight of shape `(out, in, 3, 3)` and maps `(batch, in, h, w)` to `(batch, out, h', w')`, where padding `1` with a `3×3` kernel keeps the spatial size at stride `1` and halves it (rounding up) at stride `2`. `t.nn.BatchNorm2d(out)` owns the per-channel scale, shift and running statistics of the batch-norm lesson and keeps the shape. The bias is left out of the convolution because the batch-norm that follows subtracts the channel mean, which in training mode would cancel a per-channel bias exactly (the batch norm's own shift plays that role instead).

A residual block's left branch is the chain conv → norm → ReLU → conv → norm, with the stride applied only in the first convolution. The reason the second convolution has stride `1` and `out` input channels is that the first has already done the resizing: after it, the tensor is at the block's output shape, and everything that follows works at that shape.

```python
import torch as t
conv=t.nn.Conv2d(3,8,kernel_size=3,stride=2,padding=1,
                 bias=False)
bn=t.nn.BatchNorm2d(8)
x=t.zeros(2,3,6,6)
print(conv.weight.shape, bn(conv(x)).shape)
# Hidden checks
assert conv.weight.shape==(8,3,3,3) and bn(conv(x)).shape==(2,8,3,3)
```

## Worked example

We give a one-channel `3×3` convolution all-ones weights and feed a `3×3` image of ones with padding `1`. Each output pixel counts how many of its nine neighbours lie inside the image: `9` at the centre, `6` on an edge, `4` in a corner. Predict the corner value before running.

```python
import torch as t
conv=t.nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1,
                 bias=False)
with t.no_grad():
    conv.weight.copy_(t.ones(1,1,3,3))
x=t.ones(1,1,3,3)
print(conv(x)[0,0])
# Hidden checks
assert conv(x)[0,0].detach().tolist()==[[4.,6.,4.],[6.,9.,6.],[4.,6.,4.]]
```

In evaluation mode a fresh batch-norm uses its running mean `0` and running variance `1`, so it leaves the values essentially unchanged, and the ReLU after it changes nothing because every count is positive. The left branch of a block is this convolution–norm pair twice over, with a ReLU between the two pairs and none after the second.

```python
bn=t.nn.BatchNorm2d(1)
bn.eval()
left=t.nn.Sequential(conv,bn,t.nn.ReLU())
print(left(x)[0,0,1,1])
# Hidden checks
assert abs(left(x)[0,0,1,1].item()-9.)<1e-3
```

## Faded practice

### q1389
Given channel counts `i`, `o` and a stride `s`, return the LEFT branch of a residual block as a Sequential chain of five children: a `3×3` convolution from `i` to `o` channels with stride `s`, padding 1 and no bias; a 2-D batch-norm over `o` channels; a ReLU; a `3×3` convolution from `o` to `o` channels with stride 1, padding 1 and no bias; and a second batch-norm over `o` channels. Applied to `(batch, i, h, w)` it returns `(batch, o, h', w')` with each spatial size divided by `s` and rounded up.

```python starter
import torch as t

def solve(i,o,s):
    return t.nn.Sequential(t.nn._____(i,o,kernel_size=3,stride=s,padding=1,bias=False),t.nn._____(o),t.nn.ReLU(),t.nn._____(o,o,kernel_size=3,stride=1,padding=1,bias=False),t.nn._____(o))
```

```python solution
import torch as t

def solve(i,o,s):
    return t.nn.Sequential(t.nn.Conv2d(i,o,kernel_size=3,stride=s,padding=1,bias=False),t.nn.BatchNorm2d(o),t.nn.ReLU(),t.nn.Conv2d(o,o,kernel_size=3,stride=1,padding=1,bias=False),t.nn.BatchNorm2d(o))
```

### q1390
Given a float image batch `x` of shape `(batch, c, h, w)` and a stride `s`, return the output of a `3×3` convolution layer with padding 1, stride `s`, no bias, `c` input channels and ONE output channel, whose every weight equals `1/(9*c)` — so each output pixel is the mean of the `3×3` neighbourhood across all channels, with zeros outside the image. Output shape `(batch, 1, h', w')` with each spatial size divided by `s` and rounded up.

```python starter
import torch as t

def solve(x,s):
    c=x.shape[1]
    conv=t.nn._____(c,1,kernel_size=3,stride=s,padding=1,bias=False)
    with t.no_grad():
        conv.weight.copy_(t.ones(1,c,3,3)/(9*c))
    return conv(x)
```

```python solution
import torch as t

def solve(x,s):
    c=x.shape[1]
    conv=t.nn.Conv2d(c,1,kernel_size=3,stride=s,padding=1,bias=False)
    with t.no_grad():
        conv.weight.copy_(t.ones(1,c,3,3)/(9*c))
    return conv(x)
```

## Concept: The shortcut must match the left branch, then the two add

A residual block returns `relu(left(x) + right(x))`: the left branch computes a change and the right branch, the *shortcut*, carries `x` across so the block only has to learn the difference from its input: while the left branch is small the block is close to the identity. That only adds up if both branches have the same shape. When the block keeps its shape — stride `1` and `in == out`, which is the single test `(s, i) != (1, o)` failing — the shortcut is `t.nn.Identity()`, a module whose forward returns its input untouched. When the block changes the channel count or the stride, the shortcut is a `1×1` convolution with the same stride, followed by a batch-norm, so that it resizes `x` the same way the left branch did.

The reason for a `1×1` kernel rather than `3×3` is that the shortcut is not supposed to look at neighbours; it only re-mixes channels and, through its stride, keeps every `s`-th pixel. The reason the ReLU comes after the sum rather than at the end of the left branch is that a ReLU before the sum would let the block only ever add to `x`, never subtract.

```python
import torch as t
def right_branch(i,o,s):
    if (s,i)!=(1,o):
        return t.nn.Sequential(
            t.nn.Conv2d(i,o,kernel_size=1,stride=s,
                        bias=False),
            t.nn.BatchNorm2d(o))
    return t.nn.Identity()
print(type(right_branch(4,4,1)),
      len(right_branch(4,8,2)))
# Hidden checks
assert type(right_branch(4,4,1)) is t.nn.Identity and len(right_branch(4,8,2))==2
```

## Worked example

We build a shape-preserving block on a one-channel `3×3` image of ones, with every convolution weight set to one and both batch-norms in evaluation mode. The first convolution gives the neighbour counts from before; the second convolution of that map has centre `4+6+4+6+9+6+4+6+4 = 49`. Predict the block's centre value: the shortcut adds the input's `1`.

```python
import torch as t
def left_branch(i,o,s):
    return t.nn.Sequential(
        t.nn.Conv2d(i,o,kernel_size=3,stride=s,
                    padding=1,bias=False),
        t.nn.BatchNorm2d(o),t.nn.ReLU(),
        t.nn.Conv2d(o,o,kernel_size=3,stride=1,
                    padding=1,bias=False),
        t.nn.BatchNorm2d(o))
print(len(left_branch(1,1,1)))
# Hidden checks
assert len(left_branch(1,1,1))==5
```

Five children: convolution, batch norm, ReLU, convolution, batch norm. Only the first convolution takes the stride, so only it can change the shape. The block itself is that branch beside the shortcut, added and passed through one more ReLU:

```python
class Block(t.nn.Module):
    def __init__(self,i,o,s):
        super().__init__()
        self.left=left_branch(i,o,s)
        self.right=right_branch(i,o,s)
        self.relu=t.nn.ReLU()
    def forward(self,x):
        a=self.left(x)
        b=self.right(x)
        return self.relu(a+b)
b=Block(1,1,1)
print(type(b.right))
# Hidden checks
assert type(b.right) is t.nn.Identity
```

The sum is `49 + 1 = 50` at the centre; at a corner the second convolution sees `4+6+6+9 = 25` and on an edge `4+6+4+6+9+6 = 35`, and the shortcut adds `1` to each. Every number is positive so the final ReLU is invisible here.

```python
with t.no_grad():
    for p in b.parameters():
        if p.ndim==4:
            p.copy_(t.ones(p.shape))
b.eval()
x=t.ones(1,1,3,3)
print(b(x)[0,0])
# Hidden checks
assert t.allclose(b(x)[0,0].detach(),t.tensor([[26.,36.,26.],[36.,50.,36.],[26.,36.,26.]]),atol=1e-2)
```

## Faded practice

### q1391
Given channel counts `i`, `o` and a stride `s`, return the SHORTCUT (right branch) of a residual block: an Identity module when `s` is 1 and `i` equals `o`; otherwise a Sequential chain of a `1×1` convolution from `i` to `o` channels with stride `s`, no padding and no bias, followed by a 2-D batch-norm over `o` channels.

```python starter
import torch as t

def solve(i,o,s):
    if (s,i)!=(1,o):
        return t.nn.Sequential(t.nn._____(i,o,kernel_size=1,stride=s,bias=False),t.nn._____(o))
    return t.nn._____()
```

```python solution
import torch as t

def solve(i,o,s):
    if (s,i)!=(1,o):
        return t.nn.Sequential(t.nn.Conv2d(i,o,kernel_size=1,stride=s,bias=False),t.nn.BatchNorm2d(o))
    return t.nn.Identity()
```

### q1392
Return a residual block module for channel counts `i`, `o` and stride `s` with three registered children named `left`, `right` and `relu`: `left` is the five-step branch (`3×3` conv with stride `s`, padding 1, no bias; batch-norm; ReLU; `3×3` conv stride 1, padding 1, no bias; batch-norm), `right` is the Identity when the shape is preserved and otherwise a `1×1` stride-`s` bias-free conv followed by a batch-norm, and the forward returns the ReLU of the sum of both branches.

```python starter
import torch as t

def solve(i,o,s):
    class Block(t.nn.Module):
        def __init__(self):
            super().__init__()
            self.left=t.nn.Sequential(t.nn._____(i,o,kernel_size=3,stride=s,padding=1,bias=False),t.nn._____(o),t.nn.ReLU(),t.nn._____(o,o,kernel_size=3,stride=1,padding=1,bias=False),t.nn._____(o))
            if (s,i)!=(1,o):
                self.right=t.nn.Sequential(t.nn._____(i,o,kernel_size=1,stride=s,bias=False),t.nn._____(o))
            else:
                self.right=t.nn._____()
            self.relu=t.nn.ReLU()
        def forward(self,x):
            return self.relu(self.left(x)+self.right(x))
    return Block()
```

```python solution
import torch as t

def solve(i,o,s):
    class Block(t.nn.Module):
        def __init__(self):
            super().__init__()
            self.left=t.nn.Sequential(t.nn.Conv2d(i,o,kernel_size=3,stride=s,padding=1,bias=False),t.nn.BatchNorm2d(o),t.nn.ReLU(),t.nn.Conv2d(o,o,kernel_size=3,stride=1,padding=1,bias=False),t.nn.BatchNorm2d(o))
            if (s,i)!=(1,o):
                self.right=t.nn.Sequential(t.nn.Conv2d(i,o,kernel_size=1,stride=s,bias=False),t.nn.BatchNorm2d(o))
            else:
                self.right=t.nn.Identity()
            self.relu=t.nn.ReLU()
        def forward(self,x):
            return self.relu(self.left(x)+self.right(x))
    return Block()
```

## Solo practice

### q1393
Without building any module, return the output shape of a residual block as a tuple of four Python ints: the block takes `(b, i, h, w)`, has `o` output channels and stride `s`, and every spatial size is rounded UP when divided by `s`.

### q1394
Given two modules `left` and `right` that return tensors of the same shape for `x`, return the residual combination: the ReLU of the sum of the two branch outputs, computed without defining a class.

### q1395
Given a module `f` that preserves its input's shape, return a module wrapping it as an identity-shortcut residual step: the forward returns the ReLU of the sum of the input and the output of `f`, and `f` is registered as the child named `f`.

### q1396
Given a float batch `x` of shape `(batch, c, h, w)` and a stride `s`, return the identity-shortcut downsample as a convolution: build a `1×1` convolution layer with stride `s`, no padding, no bias and `c` input and output channels, set its weight so that each output channel copies the same-index input channel, and return its output. The result equals keeping every `s`-th row and column of `x`.

### q1397
Return a residual block for channel counts `i`, `o` and stride `s` whose shortcut is ALWAYS a projection: the child `right` is a `1×1` stride-`s` bias-free convolution from `i` to `o` channels followed by a batch-norm, even when the shape is preserved. The child `left` is the usual five-step branch and `relu` is the final ReLU applied to the sum.

### q1398
Return a SHALLOW residual block for `i`, `o`, `s`: the child `left` is only conv → batch-norm (one `3×3` bias-free convolution with stride `s` and padding 1, then a batch-norm over `o` channels), `right` is the usual shortcut (Identity when shape-preserving, otherwise `1×1` stride-`s` bias-free conv plus batch-norm), and the forward returns the ReLU of the sum.

## Integrated practice

### q1399
Implement the ARENA residual block from scratch for channel counts `i`, `o` and stride `s`: children named `left` (a Sequential of `3×3` stride-`s` padding-1 bias-free conv, batch-norm, ReLU, `3×3` stride-1 padding-1 bias-free conv, batch-norm), `right` (Identity when `s` is 1 and `i` equals `o`, else a Sequential of `1×1` stride-`s` bias-free conv and batch-norm) and `relu`; forward returns the ReLU of the two branch outputs added together. Its registered tensors must appear in the order left, then right.

### q1400
Return a PRE-ACTIVATION residual block for `i`, `o`, `s`: the child `left` is batch-norm over `i` channels, ReLU, `3×3` stride-`s` padding-1 bias-free conv to `o` channels, batch-norm over `o`, ReLU, `3×3` stride-1 padding-1 bias-free conv; the child `right` is the usual shortcut (Identity when shape-preserving, else `1×1` stride-`s` bias-free conv plus batch-norm); the forward returns the plain SUM of the two branches with no ReLU after it.

### q1401
Return a BOTTLENECK residual block for input channels `i`, inner channels `mid`, output channels `o` and stride `s`: the child `left` is a Sequential of `1×1` bias-free conv `i`→`mid`, batch-norm, ReLU, `3×3` stride-`s` padding-1 bias-free conv `mid`→`mid`, batch-norm, ReLU, `1×1` bias-free conv `mid`→`o`, batch-norm; the child `right` is Identity when `s` is 1 and `i` equals `o`, else `1×1` stride-`s` bias-free conv `i`→`o` plus batch-norm; the child `relu` is applied to the sum.

## Misconceptions

- **The shortcut is always the identity.** Only when the shape is preserved; a stride or a channel change needs a `1×1` convolution and a batch-norm so the two branches can add.
- **Both convolutions in the left branch carry the stride.** Only the first does; the second works at the output shape with stride `1`.
- **The ReLU goes at the end of the left branch.** It goes after the sum; a ReLU before the sum would make the block unable to subtract from `x`.
- **A convolution followed by batch-norm needs its bias.** The norm subtracts the channel mean, which cancels the bias exactly; it is dead weight.
