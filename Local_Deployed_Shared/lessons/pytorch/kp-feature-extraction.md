---
kc: cnn.feature-extraction
title: Pretrained weights and feature extraction
new_syntax: ['Tensor.state_dict', 'builtin.zip', 'Tensor.load_state_dict', 'Tensor.parameters', 'Tensor.requires_grad_', 'Tensor.requires_grad']
concepts: [state-dict, copy-by-position, freeze-and-replace-head]
supporting: ['cnn.resnet34', 'cnn.module-state', 'cnn.linear-layer', 'cnn.sequential']
previews: []
faded: [1426, 1427, 1428, 1429, 1430, 1431]
guided: []
independent: [1432, 1433, 1434, 1435, 1436, 1437]
integrated: [1438, 1439, 1440]
---

## Concept: A module's state is an ordered mapping of named tensors

Every module can hand you its *state dict*: a mapping from a dotted name to a tensor, one entry per parameter **and** per (persistent) buffer, in registration order. A Sequential chain of a convolution and a batch norm therefore has six entries — the convolution's weight, the batch norm's weight and bias, and the batch norm's three buffers (running mean, running variance, batch counter) — even though only three of them are parameters. The names are the path through the children: `"1.running_mean"` is the `running_mean` of child `"1"`.

Two models built the same way — the same layers registered in the same order — have state dicts of the same length whose entries line up position by position, whatever the names. Equal lengths and shapes alone do not prove that: the copy is only right when you know the two registration orders match. Python's `zip` walks two sequences side by side, yielding a pair at each step, which is exactly how you line up "my name" with "their name".

```python
import torch as t
m=t.nn.Sequential(
    t.nn.Conv2d(1,2,kernel_size=3,bias=False),
    t.nn.BatchNorm2d(2))
sd=m.state_dict()
print(list(sd))
print(sd["1.running_var"])
print(list(zip(["a","b","c"],[1,2,3])))
# Hidden checks
assert list(sd)==["0.weight","1.weight","1.bias","1.running_mean","1.running_var","1.num_batches_tracked"]
assert sd["1.running_var"].tolist()==[1.,1.] and list(zip(["a","b","c"],[1,2,3]))==[("a",1),("b",2),("c",3)]
```

## Worked example

Two chains with the same layers but different child names. Predict, before running, how many entries each state dict has and which entries pair up.

```python
mine=t.nn.Sequential(
    t.nn.Conv2d(1,2,kernel_size=3,bias=False),
    t.nn.BatchNorm2d(2))
theirs=t.nn.Sequential()
theirs.add_module("conv",
                  t.nn.Conv2d(1,2,kernel_size=3,
                              bias=False))
theirs.add_module("bn",t.nn.BatchNorm2d(2))
print(len(mine.state_dict()),
      len(theirs.state_dict()))
# Hidden checks
assert len(mine.state_dict())==6==len(theirs.state_dict())
```

Six entries each, because both chains hold the same two layers — the child names differ but the count cannot. Now line the two key lists up side by side:

```python
pairs=list(zip(list(mine.state_dict()),
               list(theirs.state_dict())))
print(pairs[:2])
# Hidden checks
assert pairs[:2]==[("0.weight","conv.weight"),("1.weight","bn.weight")]
```

Six and six, and the pairing is by position: `"0.weight"` lines up with `"conv.weight"` although the names share nothing.

## Faded practice

### q1426
Given a module `m`, return a list with one `(name, shape)` tuple per entry of its state mapping, in the mapping's own order, where `shape` is a tuple of Python ints.

```python starter
import torch as t

def solve(m):
    sd=m._____()
    return [(k,tuple(sd[k].shape)) for k in sd]
```

```python solution
import torch as t

def solve(m):
    sd=m.state_dict()
    return [(k,tuple(sd[k].shape)) for k in sd]
```

### q1427
Given two modules `m` and `src` whose state mappings have equally many entries (their names, and even the shapes at a position, may differ), return a list of `(my_name, their_name, same_shape)` tuples pairing the entries of the two state mappings by position, where `same_shape` is whether the two tensors at that position have equal shapes.

```python starter
import torch as t

def solve(m,src):
    sd=m._____()
    ps=src._____()
    return [(a,b,sd[a].shape==ps[b].shape) for a,b in _____(list(sd),list(ps))]
```

```python solution
import torch as t

def solve(m,src):
    sd=m.state_dict()
    ps=src.state_dict()
    return [(a,b,sd[a].shape==ps[b].shape) for a,b in zip(list(sd),list(ps))]
```

## Concept: Copying weights by position, and what a parameter is

Loading a state dict writes every tensor in the mapping into the module's tensor of the same name. To copy a pretrained model whose names differ from yours, take your own state dict, replace each of its values with the pretrained value at the same position, and load the result back. The shapes must agree entry by entry — that is the check that the two architectures really match.

A module's *parameters* are the subset of its state that is trained: its weights and biases, not the batch-norm running statistics. Iterating a module's parameters yields those tensors in registration order, which is how you count them or inspect them.

```python
import torch as t
mine=t.nn.Sequential(
    t.nn.Conv2d(1,2,kernel_size=3,bias=False),
    t.nn.BatchNorm2d(2))
theirs=t.nn.Sequential()
theirs.add_module("conv",
                  t.nn.Conv2d(1,2,kernel_size=3,
                              bias=False))
theirs.add_module("bn",t.nn.BatchNorm2d(2))
sd=mine.state_dict()
ps=theirs.state_dict()
for a,b in zip(list(sd),list(ps)):
    sd[a]=ps[b]
mine.load_state_dict(sd)
print(len(list(mine.parameters())), len(sd))
# Hidden checks
assert (mine[0].weight-theirs[0].weight).abs().sum().item()==0
assert len(list(mine.parameters()))==3 and len(sd)==6
```

## Worked example

Fill the pretrained chain with known numbers, copy into a fresh chain, and check that both give the same output. Predict how many parameters the count reports before running.

```python
with t.no_grad():
    for p in theirs.parameters():
        p.copy_(t.linspace(
            -1.,1.,p.numel()).reshape(p.shape))
sd=mine.state_dict()
ps=theirs.state_dict()
for a,b in zip(list(sd),list(ps)):
    sd[a]=ps[b]
mine.load_state_dict(sd)
print(mine[0].weight.reshape(-1)[:3])
# Hidden checks
assert (mine[0].weight-theirs[0].weight).abs().sum().item()==0
```

The fresh chain now carries the filled numbers, so the two chains agree on any input, and the parameter count is the convolution's `18` weights plus the batch norm's `2 + 2`:

```python
x=t.arange(16.).reshape(1,1,4,4)
print(
    (mine.eval()(x)-theirs.eval()(x)).abs().sum(
    ).item(),
    sum([p.numel() for p in mine.parameters()]))
# Hidden checks
assert (mine.eval()(x)-theirs.eval()(x)).abs().sum().item()==0
assert sum([p.numel() for p in mine.parameters()])==18+2+2
```

The difference is zero and the count is `22`.

## Faded practice

### q1428
Copy the weights of `src` into `m`, two modules with the same architecture but possibly different child names: take `m`'s state mapping, replace each of its values by the value of `src`'s state mapping at the same position, load the result into `m`, and return `m`.

```python starter
import torch as t

def solve(m,src):
    sd=m._____()
    ps=src._____()
    for a,b in _____(list(sd),list(ps)):
        sd[a]=ps[b]
    m._____(sd)
    return m
```

```python solution
import torch as t

def solve(m,src):
    sd=m.state_dict()
    ps=src.state_dict()
    for a,b in zip(list(sd),list(ps)):
        sd[a]=ps[b]
    m.load_state_dict(sd)
    return m
```

### q1429
Given a module `m`, return a tuple of three Python ints: how many parameter tensors it has, how many entries its state mapping has, and the total number of values across its parameters.

```python starter
import torch as t

def solve(m):
    ps=list(m._____())
    return (len(ps), len(m._____()), sum([p.numel() for p in ps]))
```

```python solution
import torch as t

def solve(m):
    ps=list(m.parameters())
    return (len(ps), len(m.state_dict()), sum([p.numel() for p in ps]))
```

## Concept: Freeze everything, then replace the head

Feature extraction keeps a pretrained network's body fixed and trains only a new final layer. Every parameter carries a `requires_grad` flag saying whether gradients should flow into it; calling a module's `requires_grad_(False)` clears the flag on every parameter below it in one go. Replacing the last child of the head with a freshly constructed linear layer then gives you a layer whose parameters are new — and new parameters are created with the flag set — so after the two steps exactly the new head is trainable. Freezing stops gradient updates to those parameters; it does not stop a batch norm's running statistics from moving while the module is in training mode, so a frozen body is normally kept in evaluation mode as well.

The order matters: freeze first, replace second. Freezing after replacing would freeze the new head too.

```python
import torch as t
class AveragePool(t.nn.Module):
    def forward(self,x):
        return x.mean((2,3))
net=t.nn.Sequential(
    t.nn.Conv2d(3,4,kernel_size=3,padding=1,
                bias=False),t.nn.BatchNorm2d(4),
    t.nn.ReLU(),AveragePool(),t.nn.Linear(4,10))
net.requires_grad_(False)
net[-1]=t.nn.Linear(4,3)
print([p.requires_grad for p in net.parameters()])
print(net(t.zeros(2,3,8,8)).shape)
# Hidden checks
assert [p.requires_grad for p in net.parameters()]==[False,False,False,True,True]
assert net(t.zeros(2,3,8,8)).shape==(2,3)
```

## Worked example

Count trainable versus frozen parameter values after the freeze-and-replace. Predict both numbers before running: the body holds `4*3*9 + 2*4 = 116` values, the new head `4*3 + 3 = 15`.

```python
trainable=[p for p in net.parameters()
           if p.requires_grad]
frozen=[p for p in net.parameters()
        if not p.requires_grad]
print(sum([p.numel() for p in trainable]),
      sum([p.numel() for p in frozen]))
# Hidden checks
assert sum([p.numel() for p in trainable])==15 and sum([p.numel() for p in frozen])==116
```

In ResNet34 the same two lines are `requires_grad_(False)` on the whole network and a new linear layer written into `out_layers[-1]`, from `512` features to the new class count.

## Faded practice

### q1430
Return an instance of a subclass of it that builds the network with `1000` classes, freezes every parameter, and then replaces the last module of its `out_layers` child (through the children mapping by name) with a new linear layer from the last output width to `n_classes`, so only the new head is trainable. The lesson's `ResNet` class (arguments: stem width, block counts, output widths, first strides, class count) is provided.

```python starter
import torch as t

def solve(c0,n_blocks,out_feats,strides,n_classes):
    class FeatureExtractor(ResNet):
        def __init__(self):
            super().__init__(c0,n_blocks,out_feats,strides,1000)
            self._____(False)
            self._modules["out_layers"][-1]=t.nn.Linear(out_feats[-1],n_classes)
    return FeatureExtractor()
```

```python solution
import torch as t

def solve(c0,n_blocks,out_feats,strides,n_classes):
    class FeatureExtractor(ResNet):
        def __init__(self):
            super().__init__(c0,n_blocks,out_feats,strides,1000)
            self.requires_grad_(False)
            self._modules["out_layers"][-1]=t.nn.Linear(out_feats[-1],n_classes)
    return FeatureExtractor()
```

### q1431
Given a module `m`, return a tuple of two Python ints: the number of values across its trainable parameters (gradient flag set) and across its frozen parameters (flag cleared).

```python starter
import torch as t

def solve(m):
    a=sum([p.numel() for p in m._____() if p._____])
    b=sum([p.numel() for p in m._____()])-a
    return (a,b)
```

```python solution
import torch as t

def solve(m):
    a=sum([p.numel() for p in m.parameters() if p.requires_grad])
    b=sum([p.numel() for p in m.parameters()])-a
    return (a,b)
```

## Solo practice

### q1432
Given a module `m` whose children are registered in order, freeze every parameter of `m` and then make only the parameters of its LAST child trainable again; return `m`.

### q1433
Copy weights from `src` into `m` by position, but only where the two tensors at a position have the same shape, leaving every other entry of `m` unchanged; return the number of entries copied as a Python int.

### q1434
Two modules `m` and `src` have identical architecture AND identical state names. Copy from `src` into `m` only the state entries whose name starts with the string `prefix` (compare the leading characters of the name), leave the rest of `m` untouched, and return `m`.

### q1435
Given a network `m` built by the lesson's `ResNet` class (children `in_layers`, `residual_layers`, `out_layers`, the last being a Sequential chain ending in the linear head), freeze all of `m`, then replace the last module of the `out_layers` child — reached from OUTSIDE the class through the children mapping by name — with a new linear layer from that head's input width to `n`; return `m`.

### q1436
Given a module `m`, return a list with one `(child_name, trainable_values)` tuple per direct child, in registration order, where `trainable_values` is the number of values across that child's parameters whose gradient flag is set (a Python int; `0` for a child with no trainable parameters).

### q1437
Given a network `m` built by the lesson's `ResNet` class, freeze everything, then make trainable again the last `k` groups of its `residual_layers` child (k may be `0`) and the whole `out_layers` child; return `m`. Reach the children from outside the class through the children mapping by name.

## Integrated practice

### q1438
The lesson's `ResNet` class is provided, and `src` is a pretrained stand-in: a module with the same tensor sequence as a ResNet built from `c0`, `n_blocks`, `out_feats`, `strides` with `1000` classes, but possibly different child names. Build that ResNet, copy `src`'s state into it by position, freeze every parameter, replace the last module of its `out_layers` child (through the children mapping by name) with a new linear layer from the last output width to `n_classes`, and return the network.

### q1439
`m` and `src` are two modules whose state mappings hold the same tensors in the same order, under names that may match or differ (for example a ResNet and a flat Sequential chain of its layers). Copy `src`'s state into `m` by position and return a tuple: `m` itself, and the number of state entries whose names differed between the two mappings (a Python int).

### q1440
Given a network `m` from the lesson's `ResNet` class, prepare it for feature extraction with `n` classes (freeze everything, replace the last module of the `out_layers` child through the children mapping by name) and return a report tuple: the number of trainable values, the number of frozen values, the number of state entries, and the logits shape for a batch of two `3×16×16` zero images as a tuple of Python ints.

## Misconceptions

- **The state dict holds only the parameters.** It also holds buffers such as batch-norm running statistics; parameters are the trained subset.
- **Names must match to copy weights.** Loading matches by name, so you rebuild the mapping with YOUR names and THEIR values, paired by position.
- **Replace the head, then freeze.** Freezing afterwards freezes the new head too; freeze first, replace second.
- **Freezing removes the parameters.** They stay in the state dict and still run in the forward pass; only their gradient flag is cleared.
