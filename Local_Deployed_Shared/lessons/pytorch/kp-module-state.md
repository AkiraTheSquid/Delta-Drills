---
kc: cnn.module-state
title: Modules, parameters and buffers
new_syntax: ['syntax.class', 'torch.nn.Module', 'builtin.super', 'torch.nn.Parameter', 'Tensor.register_buffer']
concepts: [module, parameter, buffer]
supporting: ['python.defining-functions', 'python.dots-and-imports', 'numpy.elementwise-ufuncs', 'numpy.ndarray-model', 'tensor.row-normalization', 'torch.slice-assignment', 'numpy.dtype-astype', 'python.control-flow']
previews: []
faded: [1169, 1170, 1172, 1171]
guided: []
independent: [1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181]
integrated: [1182, 1183, 1184]
---

## Concept: A module is a callable computation

PyTorch packages a computation as a class that inherits from `t.nn.Module`. You define one method, `forward(self, x)`, saying how an input becomes an output, and then you call the instance itself — `m(x)` — rather than `m.forward(x)`. The general shape is: `class Name(t.nn.Module):`, then `def forward(self, x):` indented inside it, then `return` the result. `self` is the instance the method was called on; every method of a class receives it first.

The reason to call `m(x)` and not `forward` directly is that the call goes through `Module.__call__`, which runs framework hooks around `forward` (checks, tracing, later on device placement). Calling `forward` skips all of that, so it works today and breaks the day a hook matters. The reason to make even a stateless operation a module is uniformity: a network is a tree of modules, and a tree whose leaves are all the same kind of thing can be printed, moved and saved with one call.

```python
import torch as t
class Positive(t.nn.Module):
    def forward(self,x):
        return x.clamp(min=0)
m=Positive()
print(m(t.tensor([-3.,2.])))
# Hidden checks
assert m(t.tensor([-3.,2.])).tolist()==[0.,2.]
```

This is ReLU: positive entries pass through, negative entries become zero. It has no state, which makes it the smallest possible module.

## Worked example

We build a module that halves its input. It needs no `__init__`, because nothing is stored: `forward` is the whole definition. Predict the output before running.

```python
import torch as t
class Half(t.nn.Module):
    def forward(self,x):
        return x/2
m=Half()
print(m(t.tensor([3.,-1.])))
# Hidden checks
assert m(t.tensor([3.,-1.])).tolist()==[1.5,-0.5]
```

The same instance works on any shape, because `forward` only used elementwise arithmetic. Constructing a second instance gives an independent object of the same class; with no state, the two are indistinguishable.

```python
print(Half()(t.tensor([[8.,0.]])))
# Hidden checks
assert Half()(t.tensor([[8.,0.]])).tolist()==[[4.,0.]]
```

## Faded practice

### q1169
Return a parameter-free ReLU module. For any float tensor x, preserve positive entries and replace negative entries with zero.

```python starter
import torch as t

def solve():
    pass
```

```python solution
import torch as t

def solve():
    class ReLU(t.nn.Module):
        def forward(self,x):
            return x.clamp(min=0)
    return ReLU()
```

## Concept: Parameters are the state an optimizer learns

State that training should change is stored as a `t.nn.Parameter`, assigned to an attribute of `self` inside `__init__`. The procedure has three fixed lines: `def __init__(self):`, then `super().__init__()` as the first statement, then `self.weight = t.nn.Parameter(tensor)`. `forward` reads `self.weight` like any attribute.

`super().__init__()` runs the parent class's constructor — `Module`'s — which creates the bookkeeping that registration relies on. It must come first because the very next line, assigning a `Parameter` to `self`, is intercepted by that bookkeeping: `Module` notices a `Parameter` being attached and records it, so `m.parameters()` can later hand every trainable tensor to an optimizer. Assign before calling `super().__init__()` and there is nothing to record into; the assignment raises. A plain tensor attribute is NOT recorded — it is state the module has but the optimizer cannot see. Wrapping in `Parameter` is what makes the difference, and it also sets `requires_grad=True`, so gradients flow to it.

```python
import torch as t
class Scale(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=t.nn.Parameter(t.tensor(1.5))
    def forward(self,x):
        return x*self.weight
m=Scale()
print(m(t.tensor([2.,-4.])))
# Hidden checks
assert m(t.tensor([2.,-4.])).detach().tolist()==[3.,-6.]
```

## Worked example

We give a module a trainable offset and check that registration actually happened. First the module: the constructor stores one scalar `Parameter` named `bias`, and `forward` adds it.

```python
import torch as t
class Shift(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias=t.nn.Parameter(t.tensor(0.25))
    def forward(self,x):
        return x+self.bias
m=Shift()
print(m(t.tensor([1.,2.])))
# Hidden checks
assert m(t.tensor([1.,2.])).detach().tolist()==[1.25,2.25]
```

Now the check that matters for training: `named_parameters()` lists what an optimizer would receive. The name is the attribute name, and the tensor requires a gradient because `Parameter` set that for us.

```python
names=[name for name,_ in m.named_parameters()]
print(names, m.bias.requires_grad)
# Hidden checks
assert names==["bias"] and m.bias.requires_grad
```

## Faded practice

### q1170
Implement learned scaling: output x multiplied by weight. w: scalar float tensor. Return a new module; register its learnable scalar as weight. Its forward accepts float tensors x of arbitrary shape.

```python starter
import torch as t

def solve(w):
    pass
```

```python solution
import torch as t

def solve(w):
    class Scale(t.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=t.nn.Parameter(w.clone())
        def forward(self,x):
            return x*self.weight
    return Scale()
```

### q1172
Return an affine scalar module with optional bias. w is a scalar float tensor; b is a scalar float tensor or None. Register weight; register bias only when supplied.

```python starter
import torch as t

def solve(w,b):
    pass
```

```python solution
import torch as t

def solve(w,b):
    class Affine(t.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=t.nn.Parameter(w.clone())
            self.bias=t.nn.Parameter(b.clone()) if b is not None else None
        def forward(self,x):
            y=x*self.weight
            return y if self.bias is None else y+self.bias
    return Affine()
```

## Concept: Buffers are state that is saved but not trained

Some state belongs to a model without being learned: a calibration offset, a fixed mask, and later the running statistics of BatchNorm. Register it with `self.register_buffer("name", tensor)` inside `__init__`, after `super().__init__()`, and read it as `self.name`. The rule for choosing: if an optimizer should update it, `Parameter`; if it must travel with the model but never be trained, buffer; if neither, it is not model state at all.

The reason a buffer is more than a plain attribute is that `Module` tracks it: `m.to(dtype)` and `m.to(device)` convert buffers along with parameters, and `state_dict()` saves them, so a reloaded model has the same offset it was trained with. A plain tensor attribute is left behind by all three. And unlike a `Parameter`, a buffer is not returned by `parameters()`, so the optimizer never touches it — which is exactly the contract.

```python
import torch as t
class Offset(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("offset",t.tensor(3.))
    def forward(self,x):
        return x+self.offset
m=Offset()
print(m(t.tensor([1.,-1.])), list(m.parameters()))
# Hidden checks
assert m(t.tensor([1.,-1.])).tolist()==[4.,2.] and list(m.parameters())==[]
```

## Worked example

We confirm the two things a buffer promises: it follows the module through a dtype change, and it appears in the saved state. First the dtype: `m.to(t.float64)` converts the buffer even though it is not a parameter.

```python
m=Offset().to(t.float64)
print(m.offset.dtype)
# Hidden checks
assert m.offset.dtype==t.float64
```

Then the state dict, which is what `save` writes and `load_state_dict` reads. The buffer is in it under its registered name, and there is no parameter entry because the module has none.

```python
keys=list(m.state_dict())
print(keys)
# Hidden checks
assert keys==["offset"]
```

## Faded practice

### q1171
Return a calibrated module: subtract fixed scalar b, then multiply by learned scalar w. Store weight as a Parameter and offset as a buffer; both inputs are scalar float tensors.

```python starter
import torch as t

def solve(w,b):
    pass
```

```python solution
import torch as t

def solve(w,b):
    class Calibrated(t.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=t.nn.Parameter(w.clone())
            self.register_buffer("offset",b.clone())
        def forward(self,x):
            return (x-self.offset)*self.weight
    return Calibrated()
```

## Solo practice

### q1173
Return a parameter-free module whose forward clips any float tensor x into [0,1]: entries below zero become zero, entries above one become one.

### q1174
Return a module that multiplies x by a FIXED scalar taken from w: store it as a buffer named scale, so the module has no parameters.

### q1175
Return a module that adds a trainable scalar offset to x: register the scalar from w as a parameter named bias.

### q1176
Return a module that counts its forward calls in an integer buffer named calls (starting at zero) and returns x unchanged.

### q1177
Return a model containing one child module named scale, whose sole parameter weight starts at scalar tensor w. Output is the positive part of x times that weight. Parent parameter discovery must find scale.weight.

### q1178
Return a module with a trainable vector parameter weight initialized from w (n,), whose forward returns the weighted sum of x (...,n) along its last axis, shape (...).

### q1179
Return a learned scaling module whose parameter weight (from w) is FROZEN: it stays a registered parameter but requires no gradient. Forward returns x times weight.

### q1180
Return a module using the same trainable scalar on both sides of a ReLU: positive part of x times weight, then multiplied by weight again. Register weight only once. w: scalar float tensor.

### q1181
Return a learned scaling module (parameter weight from w, forward x times weight) whose extra_repr reports the current weight as the string "weight=<value>", using the tensor's item().

## Integrated practice

### q1182
Return a module computing weight × ReLU(weight × x), with one child named scale owning one trainable scalar weight initialized from w. Both occurrences share that parameter and accumulate gradients into it.

### q1183
Return a module with two child modules named first and second, each owning one trainable scalar parameter weight (from w and from b); forward returns second(first(x)), where each child multiplies by its weight. Gradients must reach both children.

### q1184
Return a learned scaling module (parameter weight from w) that also counts forward calls in a buffer named calls; forward returns x times weight. Its state dict must carry both, so a fresh instance loaded from it reports the same count.

## Misconceptions

- **Any tensor attribute is trainable.** Only a `t.nn.Parameter` is handed to the optimizer; a plain tensor on `self` is invisible to `parameters()`, `to()` and `state_dict()`.
- **`super().__init__()` is boilerplate that can go anywhere.** It creates the registries; a `Parameter` or buffer assigned before it has nowhere to go and raises.
- **Fixed state can be a plain attribute.** If it must move with the model or be saved, it is a buffer; otherwise it is lost on `to()` and on reload.
- **Using a parameter twice makes two copies.** Reading `self.weight` in two places shares one tensor, and both uses accumulate gradient into it; a second, independent value needs a second `Parameter`.
