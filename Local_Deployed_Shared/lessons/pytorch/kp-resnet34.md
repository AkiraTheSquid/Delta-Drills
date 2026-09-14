---
kc: cnn.resnet34
title: Assembling ResNet34
new_syntax: ['torch.nn.MaxPool2d']
concepts: [stem, stages-and-head]
supporting: ['cnn.block-group', 'cnn.pooling', 'cnn.linear-layer', 'cnn.sequential']
previews: []
faded: [1413, 1414, 1415, 1416]
guided: []
independent: [1417, 1418, 1419, 1420, 1421, 1422]
integrated: [1423, 1424, 1425]
---

## Concept: The stem shrinks the image four-fold before any block

ResNet34 opens with a *stem*: one wide convolution (`7×7` window, stride `2`, padding `3`, `3` input channels to `64`), a batch norm, a ReLU, and a max-pool with a `3×3` window, stride `2`, padding `1`. Both strided steps halve the spatial size, rounding up, so a `224×224` image reaches the first residual block as `64` channels of `56×56`. The stem is the only place in the network where a `7×7` window or a max-pool appears; everything after it is residual blocks, until the average-pool and linear head at the end.

The `padding = 3` on a `7×7` window and `padding = 1` on a `3×3` window are the "same" paddings: with stride `1` they would leave the image size unchanged, so with stride `2` each step gives exactly `ceil(size / 2)`. A max-pool module takes the same window, stride and padding arguments as a convolution but has no parameters — it keeps the largest value in each window per channel.

```python
import torch as t
stem=t.nn.Sequential(t.nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),t.nn.BatchNorm2d(64),t.nn.ReLU(),t.nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
print(stem(t.zeros(1,3,224,224)).shape)
print(t.nn.MaxPool2d(kernel_size=3,stride=2,padding=1)(t.arange(16.).reshape(1,1,4,4)))
# Hidden checks
assert stem(t.zeros(1,3,224,224)).shape==(1,64,56,56)
assert t.nn.MaxPool2d(kernel_size=3,stride=2,padding=1)(t.arange(16.).reshape(1,1,4,4)).tolist()==[[[[5.,7.],[13.,15.]]]]
```

## Worked example

We push a batch of two `3×30×30` images through a stem with `8` output channels. Predict the shape after the convolution and after the pool before running: each strided step gives `ceil(30 / 2) = 15`, then `ceil(15 / 2) = 8`.

```python
stem=t.nn.Sequential(t.nn.Conv2d(3,8,kernel_size=7,stride=2,padding=3),t.nn.BatchNorm2d(8),t.nn.ReLU(),t.nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
x=t.zeros(2,3,30,30)
print(stem[0](x).shape, stem(x).shape)
# Hidden checks
assert stem[0](x).shape==(2,8,15,15) and stem(x).shape==(2,8,8,8)
```

The pool has no parameters, so the stem's parameter count is the convolution's plus the batch norm's:

```python
print(sum([p.numel() for p in stem.parameters()]), 8*3*49+8, 2*8)
# Hidden checks
assert sum([p.numel() for p in stem.parameters()])==8*3*49+8+2*8
```

## Faded practice

### q1413
Return the ResNet stem for `c0` output channels as a Sequential chain of four modules: a convolution from `3` channels to `c0` with a `7×7` window, stride `2` and padding `3` (with bias), a batch norm over `c0` channels, a ReLU, and a max-pool with a `3×3` window, stride `2` and padding `1`.

```python starter
import torch as t

def solve(c0):
    return t.nn.Sequential(t.nn.Conv2d(3,c0,kernel_size=7,stride=2,padding=3),t.nn.BatchNorm2d(c0),t.nn.ReLU(),t.nn._____(kernel_size=3,stride=2,padding=1))
```

```python solution
import torch as t

def solve(c0):
    return t.nn.Sequential(t.nn.Conv2d(3,c0,kernel_size=7,stride=2,padding=3),t.nn.BatchNorm2d(c0),t.nn.ReLU(),t.nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
```

### q1414
Without building any module, return the shape of the stem's output for a batch `(b, 3, h, w)` and stem width `c0`, as a tuple of four Python ints. The `7×7` stride-`2` padding-`3` convolution and the `3×3` stride-`2` padding-`1` pool each halve a spatial size, rounding up.

```python starter
import torch as t

def solve(b,c0,h,w):
    return (b,_____,_____,_____)
```

```python solution
import torch as t

def solve(b,c0,h,w):
    return (b,c0,(h+3)//4,(w+3)//4)
```

## Concept: Stem, four stages, then average-pool and a linear head

The whole network is three Sequential chains in a row. `in_layers` is the stem. `residual_layers` is a chain of four block groups, with `[3, 4, 6, 3]` blocks, `[64, 128, 256, 512]` output channels and first strides `[1, 2, 2, 2]`; each group's input width is the previous group's output width, and the first group's is the stem's `64`. `out_layers` averages every channel over its remaining `7×7` spatial positions, leaving one number per channel, and a linear layer maps those `512` numbers to `n_classes` logits. Summing the block counts gives `3 + 4 + 6 + 3 = 16` blocks of two convolutions each, plus the stem convolution and the head: the "34" in the name.

Averaging over the spatial axes with `x.mean((2, 3))` turns `(batch, 512, h, w)` into `(batch, 512)` whatever `h` and `w` are, which is why the network accepts images of any spatial size, as long as every stride still leaves at least one position. It is wrapped in a tiny module so it can sit inside a Sequential chain next to the linear layer.

```python
import torch as t
class ResidualBlock(t.nn.Module):
    def __init__(self,i,o,s):
        super().__init__()
        self.left=t.nn.Sequential(t.nn.Conv2d(i,o,kernel_size=3,stride=s,padding=1,bias=False),t.nn.BatchNorm2d(o),t.nn.ReLU(),t.nn.Conv2d(o,o,kernel_size=3,stride=1,padding=1,bias=False),t.nn.BatchNorm2d(o))
        self.right=t.nn.Identity() if (s,i)==(1,o) else t.nn.Sequential(t.nn.Conv2d(i,o,kernel_size=1,stride=s,bias=False),t.nn.BatchNorm2d(o))
        self.relu=t.nn.ReLU()
    def forward(self,x):
        return self.relu(self.left(x)+self.right(x))
class BlockGroup(t.nn.Module):
    def __init__(self,n,i,o,s):
        super().__init__()
        self.blocks=t.nn.Sequential()
        self.blocks.add_module("0",ResidualBlock(i,o,s))
        k=1
        for extra in [o]*(n-1):
            self.blocks.add_module(str(k),ResidualBlock(o,o,1))
            k+=1
    def forward(self,x):
        return self.blocks(x)
# Hidden checks
assert BlockGroup(2,4,8,2)(t.zeros(1,4,8,8)).shape==(1,8,4,4)
```

```python
class AveragePool(t.nn.Module):
    def forward(self,x):
        return x.mean((2,3))
class ResNet(t.nn.Module):
    def __init__(self,c0,n_blocks,out_feats,strides,n_classes):
        super().__init__()
        self.in_layers=t.nn.Sequential(t.nn.Conv2d(3,c0,kernel_size=7,stride=2,padding=3),t.nn.BatchNorm2d(c0),t.nn.ReLU(),t.nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
        self.residual_layers=t.nn.Sequential()
        i=c0
        k=0
        for n in n_blocks:
            self.residual_layers.add_module(str(k),BlockGroup(n,i,out_feats[k],strides[k]))
            i=out_feats[k]
            k+=1
        self.out_layers=t.nn.Sequential(AveragePool(),t.nn.Linear(out_feats[-1],n_classes))
    def forward(self,x):
        return self.out_layers(self.residual_layers(self.in_layers(x)))
# Hidden checks
assert ResNet(8,[1,1],[8,16],[1,2],5)(t.zeros(2,3,32,32)).shape==(2,5)
```

## Worked example

A scaled-down network: stem width `8`, two groups of one block with widths `[8, 16]` and strides `[1, 2]`, `5` classes. For a `3×32×32` image the stem gives `8×8×8`, the first group keeps `8×8`, the second halves it to `4×4` with `16` channels, the average-pool leaves `16` numbers, and the head gives `5` logits. Predict the three intermediate shapes before running.

```python
net=ResNet(8,[1,1],[8,16],[1,2],5)
x=t.zeros(2,3,32,32)
a=net.in_layers(x)
b=net.residual_layers(a)
print(a.shape, b.shape, net.out_layers(b).shape)
# Hidden checks
assert a.shape==(2,8,8,8) and b.shape==(2,16,4,4) and net.out_layers(b).shape==(2,5)
```

The real ResNet34 is the same class with the paper's numbers, and its last feature map for a `224×224` input is `512` channels of `7×7`:

```python
big=ResNet(64,[3,4,6,3],[64,128,256,512],[1,2,2,2],1000)
print(sum([len(g.blocks) for g in big.residual_layers]), big.residual_layers(big.in_layers(t.zeros(1,3,224,224))).shape)
# Hidden checks
assert sum([len(g.blocks) for g in big.residual_layers])==16
assert big.residual_layers(big.in_layers(t.zeros(1,3,224,224))).shape==(1,512,7,7)
```

## Faded practice

### q1415
Return a ResNet module with three children: `in_layers` (the stem for `c0` channels: `7×7` stride-`2` padding-`3` convolution from `3` channels with bias, batch norm, ReLU, `3×3` stride-`2` padding-`1` max-pool), `residual_layers` (a Sequential chain of one group per entry of `n_blocks`, with the matching entries of `out_feats` and `strides`, each group's input width being the previous group's output width and `c0` for the first), and `out_layers` (a module averaging over the two spatial axes followed by a linear layer from the last output width to `n_classes`). The forward applies the three in order. Classes `ResidualBlock` and `BlockGroup` (arguments: block count, in channels, out channels, first stride) are provided.

```python starter
import torch as t

def solve(c0,n_blocks,out_feats,strides,n_classes):
    class AveragePool(t.nn.Module):
        def forward(self,x):
            return x.mean((2,3))
    class ResNet(t.nn.Module):
        def __init__(self):
            super().__init__()
            self.in_layers=t.nn.Sequential(t.nn.Conv2d(3,c0,kernel_size=7,stride=2,padding=3),t.nn.BatchNorm2d(c0),t.nn.ReLU(),t.nn._____(kernel_size=3,stride=2,padding=1))
            self.residual_layers=t.nn.Sequential()
            i=c0
            k=0
            for n in n_blocks:
                self.residual_layers.add_module(str(k),BlockGroup(_____,_____,_____,_____))
                i=_____
                k+=1
            self.out_layers=t.nn.Sequential(AveragePool(),t.nn.Linear(out_feats[-1],n_classes))
        def forward(self,x):
            return self.out_layers(self.residual_layers(self.in_layers(x)))
    return ResNet()
```

```python solution
import torch as t

def solve(c0,n_blocks,out_feats,strides,n_classes):
    class AveragePool(t.nn.Module):
        def forward(self,x):
            return x.mean((2,3))
    class ResNet(t.nn.Module):
        def __init__(self):
            super().__init__()
            self.in_layers=t.nn.Sequential(t.nn.Conv2d(3,c0,kernel_size=7,stride=2,padding=3),t.nn.BatchNorm2d(c0),t.nn.ReLU(),t.nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
            self.residual_layers=t.nn.Sequential()
            i=c0
            k=0
            for n in n_blocks:
                self.residual_layers.add_module(str(k),BlockGroup(n,i,out_feats[k],strides[k]))
                i=out_feats[k]
                k+=1
            self.out_layers=t.nn.Sequential(AveragePool(),t.nn.Linear(out_feats[-1],n_classes))
        def forward(self,x):
            return self.out_layers(self.residual_layers(self.in_layers(x)))
    return ResNet()
```

### q1416
Return the classification head as a module: given a feature map `(batch, c, h, w)` it averages every channel over the two spatial axes to `(batch, c)` and applies a linear layer (child `lin`) from `c` to `n_classes`, returning `(batch, n_classes)` logits.

```python starter
import torch as t

def solve(c,n_classes):
    class Head(t.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin=t.nn.Linear(_____,_____)
        def forward(self,x):
            return self.lin(_____)
    return Head()
```

```python solution
import torch as t

def solve(c,n_classes):
    class Head(t.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin=t.nn.Linear(c,n_classes)
        def forward(self,x):
            return self.lin(x.mean((2,3)))
    return Head()
```

## Solo practice

### q1417
Without building any module, return the shape of the feature map that leaves the residual stages, for an input batch `(b, 3, h, w)`, group output widths `out_feats` and group first strides `strides`, as a tuple of four Python ints. The stem quarters each spatial size (rounding up at each of its two halvings), then each group divides by its first stride, rounding up.

### q1418
Two halvings throw away most of a small image's detail, so return a small-image stem for `c0` output channels: a Sequential chain of a convolution from `3` channels to `c0` with a `3×3` window, stride `1`, padding `1` and bias, a batch norm over `c0` channels, and a ReLU — no pooling, so the spatial size is unchanged.

### q1419
Return a feature network with children `in_layers` (the standard stem for `c0` channels: `7×7` stride-`2` padding-`3` convolution from `3` channels with bias, batch norm, ReLU, `3×3` stride-`2` padding-`1` max-pool) and `residual_layers` (the chained groups from `n_blocks`, `out_feats`, `strides`, input widths chained from `c0`), and NO head: the forward returns the spatially averaged features, shape `(batch, last output width)`. Classes `ResidualBlock` and `BlockGroup` (block count, in channels, out channels, first stride) are provided.

### q1420
Without building any module, return the parameter count of the stem plus the head as a Python int: the stem is a `7×7` convolution from `3` channels to `c0` channels WITH bias followed by a batch norm (which has an affine weight and bias per channel) and a parameter-free pool; the head is a linear layer from `c` features to `n_classes` outputs with bias. Count every weight and bias value.

### q1421
Given a ResNet-style module `net` already in evaluation mode and an image batch `x` of shape `(batch, 3, h, w)`, return the predicted class index of every image: the position of the largest logit along the class axis, as a 1-D integer tensor of length `batch`.

### q1422
Return the stem for images with `cin` channels (for example `1` for grayscale) and `c0` output channels: a Sequential chain of a `7×7` stride-`2` padding-`3` convolution with bias from `cin` to `c0`, a batch norm over `c0` channels, a ReLU, and a `3×3` stride-`2` padding-`1` max-pool.

## Integrated practice

### q1423
Return a ResNet module for images with `cin` channels: children `in_layers` (stem from `cin` to `c0`: `7×7` stride-`2` padding-`3` convolution with bias, batch norm, ReLU, `3×3` stride-`2` padding-`1` max-pool), `residual_layers` (groups from `n_blocks`, `out_feats`, `strides`, widths chained from `c0`) and `head` (a linear layer from the last width to `n_classes`). The forward returns a tuple: the spatially averaged features of shape `(batch, last width)`, and the logits of shape `(batch, n_classes)` computed from them. Classes `ResidualBlock` and `BlockGroup` (block count, in channels, out channels, first stride) are provided.

### q1424
Without building any module, return the total parameter count of the ResNet from the lesson (stem, groups, head) for the given configuration, as a Python int. Stem: a `7×7` convolution from `3` to `c0` channels with bias, plus a batch norm (an affine weight and bias per channel). Each residual block from `a` to `b` channels: a `3×3` bias-free convolution from `a` to `b`, then a `3×3` bias-free convolution from `b` to `b`, each followed by a batch norm, plus a `1×1` bias-free projection from `a` to `b` with its own batch norm exactly when the block's stride is not 1 or `a` differs from `b`. Only a group's first block resizes. Head: `last * n_classes + n_classes`.

### q1425
Return a ResNet module (children `in_layers`, `residual_layers`, `out_layers` exactly as in the lesson: stem from `3` channels to `c0`, chained groups, averaging module plus linear head to `n_classes`) whose forward returns the LIST of shapes seen along the way as tuples of Python ints: the input shape, the shape after the stem, the shape after each group in turn, and the final logits shape. Classes `ResidualBlock` and `BlockGroup` (block count, in channels, out channels, first stride) are provided.

## Misconceptions

- **The stem's max-pool has weights.** It has none; the stem's parameters are the convolution's and the batch norm's.
- **Every group starts from `64` channels.** Only the first; each group's input width is the previous group's output width.
- **The head flattens the `7×7` map into `512 * 49` features.** It averages each channel to one number, so the head is `512 → n_classes` and the input size is free.
- **"34" counts the residual blocks.** It counts convolutions plus the head: `16` blocks of two, one stem convolution, one linear layer.
