---
kc: cnn.block-group
title: A group of residual blocks
new_syntax: []
concepts: [first-block-resizes]
supporting: ['cnn.residual-block', 'cnn.sequential', 'python.control-flow']
previews: []
faded: [1402, 1403]
guided: []
independent: [1404, 1405, 1406, 1407, 1408, 1409]
integrated: [1410, 1411, 1412]
---

## Concept: Only the first block of a group resizes

A ResNet is built from *groups* of residual blocks. Within a group every block has the same output channel count `o`, and the whole group's change of shape happens in its first block: that block takes the group's input channels `i` and the group's stride `s`, so it is the only block that can need a projection shortcut (it needs one exactly when `s` is not `1` or `i` differs from `o`); every later block is `(o, o, 1)` — same channels in and out, stride `1`, identity shortcut. A group of `n` blocks is therefore a chain of one resizing block followed by `n - 1` shape-preserving ones, and its output is `(batch, o, h', w')` with the spatial size divided by `s` once, rounded up.

The reason the resizing is done once, up front, is that the group is meant to work at one resolution: the first block moves the tensor to that resolution and channel count, and the remaining blocks refine it there. Doing the stride in every block would shrink the image `n` times; and once the first block has produced `o` channels every later block must accept `o` — a later block built for `i` inputs would fail on a channel mismatch whenever `i` differs from `o`.

```python
import torch as t
class Block(t.nn.Module):
    def __init__(self,i,o,s):
        super().__init__()
        self.left=t.nn.Sequential(
            t.nn.Conv2d(i,o,kernel_size=3,stride=s,
                        padding=1,bias=False),
            t.nn.BatchNorm2d(o),t.nn.ReLU(),
            t.nn.Conv2d(o,o,kernel_size=3,stride=1,
                        padding=1,bias=False),
            t.nn.BatchNorm2d(o))
        self.right=t.nn.Identity() if (s,i)==(1,
            o) else t.nn.Sequential(
            t.nn.Conv2d(i,o,kernel_size=1,stride=s,
                        bias=False),
            t.nn.BatchNorm2d(o))
        self.relu=t.nn.ReLU()
    def forward(self,x):
        a=self.left(x)
        b=self.right(x)
        return self.relu(a+b)
print(Block(4,8,2)(t.zeros(1,4,6,6)).shape)
# Hidden checks
assert Block(4,8,2)(t.zeros(1,4,6,6)).shape==(1,8,3,3)
```

## Worked example

We build a group of three blocks that takes `4` channels to `8` at stride `2`. The first block is `Block(4, 8, 2)`; the other two are `Block(8, 8, 1)`. Predict how many of the three blocks have an Identity shortcut before running.

```python
class Group(t.nn.Module):
    def __init__(self,n,i,o,s):
        super().__init__()
        self.blocks=t.nn.Sequential()
        self.blocks.add_module("0",Block(i,o,s))
        k=1
        for extra in [o]*(n-1):
            self.blocks.add_module(str(k),
                                   Block(o,o,1))
            k+=1
    def forward(self,x):
        return self.blocks(x)
g=Group(3,4,8,2)
print(len(g.blocks),
      [type(b.right) is t.nn.Identity
        for b in g.blocks])
# Hidden checks
assert len(g.blocks)==3 and [type(b.right) is t.nn.Identity for b in g.blocks]==[False,True,True]
```

Two of the three: only the first block changes shape, so only the first needs a projection. A `6×6` input comes out `3×3` with `8` channels, halved exactly once.

```python
x=t.zeros(2,4,6,6)
print(g(x).shape)
# Hidden checks
assert g(x).shape==(2,8,3,3)
```

## Faded practice

### q1402
Return a block-group module for `n` blocks, input channels `i`, output channels `o` and first stride `s`: its single child `blocks` is a Sequential chain of `n` residual blocks registered under the names "0", "1", …, where only the first block changes the channel count (from `i` to `o`) and applies the stride `s`, and every later block keeps `o` channels at stride 1; the forward applies the chain. A class `ResidualBlock` taking (in channels, out channels, stride) is provided.

```python starter
import torch as t

def solve(n,i,o,s):
    class BlockGroup(t.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks=t.nn.Sequential()
            self.blocks.add_module("0",ResidualBlock(_____,_____,_____))
            k=1
            for extra in [o]*(n-1):
                self.blocks.add_module(str(k),ResidualBlock(_____,_____,_____))
                k+=1
        def forward(self,x):
            return self.blocks(x)
    return BlockGroup()
```

```python solution
import torch as t

def solve(n,i,o,s):
    class BlockGroup(t.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks=t.nn.Sequential()
            self.blocks.add_module("0",ResidualBlock(i,o,s))
            k=1
            for extra in [o]*(n-1):
                self.blocks.add_module(str(k),ResidualBlock(o,o,1))
                k+=1
        def forward(self,x):
            return self.blocks(x)
    return BlockGroup()
```

### q1403
Without building any module, return the list of `(in, out, stride)` argument triples for the `n` blocks of a group with input channels `i`, output channels `o` and first stride `s`: the first triple is `(i, o, s)` and each of the other `n - 1` triples is `(o, o, 1)`. Return a list of tuples.

```python starter
import torch as t

def solve(n,i,o,s):
    return [(_____,_____,_____)]+[(_____,_____,_____)]*(n-1)
```

```python solution
import torch as t

def solve(n,i,o,s):
    return [(i,o,s)]+[(o,o,1)]*(n-1)
```

## Solo practice

### q1404
Given the first group's input channel count `c0` and three equal-length lists `n_blocks`, `out_feats` and `strides`, return the list of `(n, i, o, s)` argument tuples for building each group in order: `i` is `c0` for the first group and the previous group's `o` for every later group.

### q1405
Without building any module, return the output shape of a group of `n` residual blocks applied to `(b, i, h, w)`, with output channels `o` and first stride `s`, as a tuple of four Python ints. Spatial sizes divide by the stride once and round up.

### q1406
Return a group module whose child `blocks` chains `n` blocks with the stride in the LAST block instead of the first: the first block takes `i` channels to `o` at stride 1, the middle ones keep `o` channels at stride 1, and the last block keeps `o` channels at stride `s`. When there is only one block it takes `i` to `o` at stride `s`. The forward applies the chain. A class `ResidualBlock` taking (in channels, out channels, stride) is provided.

### q1407
Return a group module whose child `blocks` applies ONE shared shape-preserving block (channels `o` to `o`, stride 1) `n` times in a row (the same block object registered `n` times under "0", "1", …), so the group has exactly one block's worth of parameters. The forward applies the chain. A class `ResidualBlock` taking (in channels, out channels, stride) is provided.

### q1408
Without building any module, return the number of trainable parameters in a group of `n` residual blocks with input channels `i`, output channels `o` and first stride `s`, as a Python int. Count: each `3×3` bias-free convolution from `a` to `b` channels has `b*a*9` weights, each batch-norm over `b` channels has `2*b` (scale and shift), and a projection shortcut (`1×1` conv `i`→`o` plus batch-norm) is present in the first block exactly when the stride is not 1 or `i` differs from `o`.

### q1409
Return a group module whose child `blocks` chains `n` blocks that DOUBLE the channels at every block: block `k` maps `i*2**k` channels to `i*2**(k+1)`, and only the first block uses stride `s` (the rest stride 1). The forward applies the chain. A class `ResidualBlock` taking (in channels, out channels, stride) is provided.

## Integrated practice

### q1410
Return a module with two children `g1` and `g2`, each a Sequential chain of residual blocks: `g1` has `n1` blocks taking `i` channels to `o1` with first stride `s1`; `g2` has `n2` blocks taking `o1` channels to `o2` with first stride `s2`. In each chain only the first block resizes and the rest are `(o, o, 1)`. The forward applies `g1` then `g2`. A class `ResidualBlock` taking (in channels, out channels, stride) is provided.

### q1411
Given the first group's input channel count `c0` and equal-length lists `n_blocks`, `out_feats`, `strides`, return one Sequential chain of GROUPS, each group itself a Sequential chain of residual blocks: group number `k` has the `k`-th entry of `n_blocks` as its block count, of `out_feats` as its output channels and of `strides` as its first stride, and input channels equal to `c0` for the first group and the previous group's output channels after that. Within a group only the first block resizes. A class `ResidualBlock` taking (in channels, out channels, stride) is provided.

### q1412
Return a group module (child `blocks`: a resizing block taking `i` channels to `o` at stride `s`, followed by the remaining shape-preserving blocks with `o` channels at stride 1, `n` blocks in all) whose forward returns the LIST of outputs after each block, so the list has `n` tensors and its last entry is the group's usual output. A class `ResidualBlock` taking (in channels, out channels, stride) is provided.

## Misconceptions

- **Every block in a group carries the stride.** Only the first; a stride in each of `n` blocks would shrink the image `n` times.
- **Every block in a group maps `i` to `o`.** Only the first takes `i`; the rest are `(o, o, 1)`.
- **The group needs its own ReLU or norm after the blocks.** Each block already ends in a ReLU; the group is just the chain.
- **A group with `n = 1` is a plain block.** It is a chain of one resizing block — still wrapped, so groups compose uniformly.
