---
kc: cnn.sequential
title: Chaining modules in order
new_syntax: ['Tensor.add_module', 'Tensor._modules', 'torch.nn.Sequential', 'torch.nn.ReLU', 'torch.nn.Linear', 'Tensor.weight']
concepts: [registration-order, library-chain, layer-objects]
supporting: ['cnn.module-state', 'cnn.batch-normalization', 'python.control-flow', 'python.lists-and-tuples']
previews: []
faded: [1374, 1375, 1376, 1377, 1378, 1379]
guided: []
independent: [1380, 1381, 1382, 1383, 1384, 1385, 423]
integrated: [1386, 1387, 1388]
---

## Concept: Registered children run in registration order

A module can own other modules the way it owns parameters. `self.add_module(name, mod)` registers `mod` as a child under a string name, and the module keeps every child in `self._modules`, a dictionary from name to child that remembers the order of insertion. A chain's `forward` walks the names in `self._modules` in that order, looks each child up and feeds each output into the next input, so the first module registered is the first one applied. Registering by name is the one mechanism behind both `self.layer = mod` and `add_module`; the explicit form is what you reach for when the children arrive in a list and their names are just their positions.

The reason to register rather than keep a plain Python list is the same reason a `Parameter` beats a plain tensor: only registered children are visible to `parameters()`, `state_dict()` and `to()`. A list attribute holding modules would compute correctly and then hand the optimizer nothing to train.

```python
import torch as t
class Double(t.nn.Module):
    def forward(self,x):
        return x*2
class AddOne(t.nn.Module):
    def forward(self,x):
        return x+1
class Chain(t.nn.Module):
    def __init__(self,mods):
        super().__init__()
        i=0
        for mod in mods:
            self.add_module(str(i),mod)
            i+=1
    def forward(self,x):
        for name in self._modules:
            x=self._modules[name](x)
        return x
print(list(Chain([Double(),AddOne()])._modules))
# Hidden checks
assert list(Chain([Double(),AddOne()])._modules)==["0","1"]
```

## Worked example

We chain `Double` then `AddOne` and apply the result to `3`. Registration order is application order, so the value is doubled first and then incremented: `3*2+1 = 7`. Predict what the reversed chain gives before running the second cell.

```python
m=Chain([Double(),AddOne()])
print(list(m._modules), m(t.tensor(3.)))
# Hidden checks
assert list(m._modules)==["0","1"] and m(t.tensor(3.)).item()==7.
```

Reversing the list reverses the computation: `(3+1)*2 = 8`. Same two children, different order, different function — the chain is the order, not the set.

```python
r=Chain([AddOne(),Double()])
print(r(t.tensor(3.)))
# Hidden checks
assert r(t.tensor(3.)).item()==8.
```

## Faded practice

### q1374
Return a module that applies each module in the list `mods` in order, feeding every output into the next module. Register the children under the names "0", "1", … so that the module's registered children come out in application order. For an empty list the forward returns its input unchanged.

```python starter
import torch as t

def solve(mods):
    class Chain(t.nn.Module):
        def __init__(self):
            super().__init__()
            i=0
            for mod in mods:
                self._____(str(i),mod)
                i+=1
        def forward(self,x):
            for name in self._____:
                x=self._____[name](x)
            return x
    return Chain()
```

```python solution
import torch as t

def solve(mods):
    class Chain(t.nn.Module):
        def __init__(self):
            super().__init__()
            i=0
            for mod in mods:
                self.add_module(str(i),mod)
                i+=1
        def forward(self,x):
            for name in self._modules:
                x=self._modules[name](x)
            return x
    return Chain()
```

### q1375
Without building a module, return the result of applying the modules in the list `mods` to the tensor `x` in order: the first module sees `x`, each later module sees the previous output. An empty list returns `x`.

```python starter
import torch as t

def solve(mods,x):
    for mod in mods:
        x=_____
    return _____
```

```python solution
import torch as t

def solve(mods,x):
    for mod in mods:
        x=mod(x)
    return x
```

## Concept: The library chain is `t.nn.Sequential`

`t.nn.Sequential(a, b, c)` is the ready-made version of that chain: its children are registered under `"0"`, `"1"`, `"2"` in the order given, `forward` applies them in that order, `len(m)` counts them and `m[i]` returns the `i`-th child, with negative indices counting from the end like a list. An empty `t.nn.Sequential()` is a chain with no steps whose forward returns its input, and `add_module` still works on it, so a chain can be built up in a loop. A chain can also be walked directly: `for mod in m` visits the children in order.

A step in a chain does not need parameters. `t.nn.ReLU()` is a module whose forward is `x.clamp(min=0)`, and packaging the nonlinearity as a module is what lets it sit between two layers as an ordinary step rather than as a special case inside somebody's `forward`.

```python
import torch as t
class Double(t.nn.Module):
    def forward(self,x):
        return x*2
class AddOne(t.nn.Module):
    def forward(self,x):
        return x+1
m=t.nn.Sequential(Double(),t.nn.ReLU(),AddOne())
print(len(m), m[1], m[-1])
# Hidden checks
assert len(m)==3 and isinstance(m[1],t.nn.ReLU) and isinstance(m[-1],AddOne)
```

## Worked example

We push `[-1, 2]` through the chain. Doubling gives `[-2, 4]`, the ReLU clamps the negative entry to `0`, and adding one gives `[1, 5]`. Predict what happens to `-1` if the ReLU is removed.

```python
x=t.tensor([-1.,2.])
print(m(x))
# Hidden checks
assert m(x).tolist()==[1.,5.]
```

Without the clamp the negative entry survives: `-2+1 = -1`. The two chains agree on the positive entry and disagree on the negative one, which is exactly the nonlinearity's job.

```python
n=t.nn.Sequential()
n.add_module("0",Double())
n.add_module("1",AddOne())
print(n(x), len(n))
# Hidden checks
assert n(x).tolist()==[-1.,5.] and len(n)==2
```

## Faded practice

### q1376
Return a library Sequential chain that applies the modules in the list `mods` in order with a fresh ReLU module inserted immediately after EACH of them, so the chain has twice as many children as `mods` has entries. Start from an empty chain and register the children under the names "0", "1", … in application order.

```python starter
import torch as t

def solve(mods):
    seq=t.nn._____()
    i=0
    for mod in mods:
        seq._____(str(i),mod)
        seq._____(str(i+1),t.nn._____())
        i+=2
    return seq
```

```python solution
import torch as t

def solve(mods):
    seq=t.nn.Sequential()
    i=0
    for mod in mods:
        seq.add_module(str(i),mod)
        seq.add_module(str(i+1),t.nn.ReLU())
        i+=2
    return seq
```

### q1377
Given a Sequential chain `m`, a tensor `x` and an integer `k` between 0 and the number of children, return the result of applying only the FIRST `k` children of `m` to `x` in order. A `k` of 0 returns `x` itself; a `k` equal to the number of children gives the same as the full chain.

```python starter
import torch as t

def solve(m,x,k):
    for mod in _____:
        x=_____
    return x
```

```python solution
import torch as t

def solve(m,x,k):
    for mod in list(m)[:k]:
        x=mod(x)
    return x
```

## Concept: A layer is an object with its weight inside

`t.nn.Linear(in_features, out_features)` is the affine map from the linear-layer lesson packaged as a module: it owns `.weight`, a parameter of shape `(out_features, in_features)`, and `.bias` of shape `(out_features,)`, both filled with random values at construction, and its forward computes `x @ weight.T + bias` on the last axis. Passing `bias=False` leaves the bias out. Because the weight is a parameter, setting it by hand is an in-place write under `t.no_grad()`, the same way the batch-norm buffers were updated.

The reason the layer carries its own weight is that a chain has nothing to pass alongside `x`: each step receives one tensor and returns one tensor, so any state a step needs must live inside the step. That is what makes `Sequential(Linear, ReLU, Linear)` a complete network rather than a recipe that still wants its weights handed in.

```python
import torch as t
lin=t.nn.Linear(2,1,bias=False)
print(lin.weight.shape)
with t.no_grad():
    lin.weight.copy_(t.tensor([[1.,-1.]]))
print(lin(t.tensor([3.,1.])))
# Hidden checks
assert lin.weight.shape==(1,2) and lin(t.tensor([3.,1.])).tolist()==[2.]
```

## Worked example

We build a one-layer network with a ReLU after it and feed two examples. The layer computes `3-1 = 2` for the first row and `1-3 = -2` for the second; the ReLU keeps `2` and clamps `-2` to `0`. Predict the output shape before running.

```python
net=t.nn.Sequential(lin,t.nn.ReLU())
x=t.tensor([[3.,1.],[1.,3.]])
print(net(x))
# Hidden checks
assert net(x).detach().tolist()==[[2.],[0.]]
```

The weight is a real parameter: it requires a gradient and it is the only entry the chain would hand an optimizer. The ReLU contributes nothing, which is why the count is one.

```python
print(net[0].weight.requires_grad, len(list(net.state_dict())))
# Hidden checks
assert net[0].weight.requires_grad and len(list(net.state_dict()))==1
```

## Faded practice

### q1378
Given a 2-D float tensor `w` of shape `(out, in)`, return a library Linear layer with no bias whose weight parameter equals `w`. Its forward maps a tensor of shape `(..., in)` to `(..., out)` by multiplying with the transpose of `w`.

```python starter
import torch as t

def solve(w):
    lin=t.nn._____(w.shape[1],w.shape[0],bias=False)
    with t.no_grad():
        lin._____.copy_(w)
    return lin
```

```python solution
import torch as t

def solve(w):
    lin=t.nn.Linear(w.shape[1],w.shape[0],bias=False)
    with t.no_grad():
        lin.weight.copy_(w)
    return lin
```

### q1379
Given `w1` of shape `(hidden, in)` and `w2` of shape `(out, hidden)`, return a Sequential chain of three children: a bias-free Linear layer whose weight is `w1`, a ReLU module, and a bias-free Linear layer whose weight is `w2`. Applied to a `(batch, in)` input it returns `(batch, out)`.

```python starter
import torch as t

def solve(w1,w2):
    a=t.nn._____(w1.shape[1],w1.shape[0],bias=False)
    b=t.nn._____(w2.shape[1],w2.shape[0],bias=False)
    with t.no_grad():
        a._____.copy_(w1)
        b._____.copy_(w2)
    return t.nn._____(a,t.nn._____(),b)
```

```python solution
import torch as t

def solve(w1,w2):
    a=t.nn.Linear(w1.shape[1],w1.shape[0],bias=False)
    b=t.nn.Linear(w2.shape[1],w2.shape[0],bias=False)
    with t.no_grad():
        a.weight.copy_(w1)
        b.weight.copy_(w2)
    return t.nn.Sequential(a,t.nn.ReLU(),b)
```

## Solo practice

### q1380
Given a Sequential chain `m`, return a NEW chain of the same library type containing the same child modules in reverse order, registered under the names "0", "1", … . `m` itself must be left unchanged.

### q1381
Given a module `mod` and an integer `k` of at least 1, return a Sequential chain that applies `mod` `k` times in a row: the same module object registered `k` times under the names "0" … "k-1".

### q1382
Given a Sequential chain `m`, return the number of its direct children whose type is exactly the library ReLU module, as a Python int.

### q1383
Given a Sequential chain `m` that contains at least one library Linear layer among its direct children, return the weight tensor of the LAST such child, shape `(out, in)`.

### q1384
Given a Sequential chain `m`, an index `i` (possibly negative), a module `mod` and a tensor `x`: replace the child at position `i` of `m` by `mod` using index assignment, and return the output of the modified chain for `x`.

### q1385
Given a Sequential chain `m` and a tensor `x`, return the list of intermediate results: element `i` is the tensor after the first `i+1` children have been applied, so the list has one entry per child and its last entry equals the full chain's output. Return an empty list for an empty chain.

### q423
Subclass the base class, register one affine layer as an attribute, apply it in forward — return an instance.

## Integrated practice

### q1386
Implement your own Sequential: return a module built from the list `mods` whose children are registered under the names "0", "1", … in order, whose forward chains them, which supports the built-in length function and integer indexing with ordinary sequence bounds (negative indices count from the end; an index past either end raises the usual IndexError), and whose registered parameters include every child's parameters. The library Sequential class is off limits; subclass the base module class directly.

### q1387
Given a list `ws` of 2-D float weight tensors where weight number `i` has shape `(out_i, in_i)` and each `in` matches the previous `out`, return a Sequential multilayer perceptron: one bias-free Linear layer per weight, in list order, with a ReLU module between consecutive layers and none after the last. A single weight gives a chain of one layer.

### q1388
Return a module that chains the modules in `mods` in order (registered under the names "0", "1", …) and ALSO counts its forward calls: it registers an integer buffer named `calls` starting at 0 that grows by one on every forward, so reading that buffer gives the number of times the module has been applied. The count is a buffer, not a parameter.

## Misconceptions

- **A Python list of modules is a chain.** Only registered children are seen by `parameters()` and `state_dict()`; a list attribute computes but cannot be trained or saved.
- **`Sequential` applies modules to the original input.** Each step receives the previous step's output; only the first sees `x`.
- **A `ReLU` module has parameters.** It is a parameter-free step; it exists as a module so it can sit in a chain.
- **`t.nn.Linear(a, b)` has a weight of shape `(a, b)`.** The weight is `(out, in)`, so `(b, a)`; the forward multiplies by its transpose.
