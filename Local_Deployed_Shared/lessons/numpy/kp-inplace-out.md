---
kc: numpy.inplace-out
title: In-place operations and the trailing underscore
supporting: [numpy.elementwise-ufuncs, numpy.slicing-views, numpy.fancy-indexing]
new_syntax: [Tensor.add_, Tensor.copy_, Tensor.div_, Tensor.mul_, Tensor.neg_, Tensor.sort]
faded: [235, 138, 59]
guided: []
independent: []
---

## Concept: the trailing underscore

Most PyTorch expressions allocate a fresh tensor per step. Usually fine — but
"in place" tasks (and memory-tight code) need the alternatives.

PyTorch marks in-place operations with a **trailing underscore**: `add_`,
`mul_`, `div_`, `neg_`, `clamp_`, `copy_`. Every one of them writes into the
existing buffer and returns that same tensor. Augmented operators (`x += 1`,
`x *= 2`) are the operator spelling of the same thing, where their written-out
forms (`x = x + 1`) allocate a new tensor and rebind the name.

The underscore is the whole signal, and it is worth trusting: a method without
one **never** modifies its receiver. `x.sort()` returns a sorted copy and
leaves `x` alone — there is no `sort_`, so sorting a tensor in place means
copying the sorted values back with `x.copy_(...)`.

The mirror-image rule from the slicing KP still applies: "do not modify the
input" → clone first. This KP is the deliberate OPPOSITE — recognize which
contract a task states before choosing tools.

## Worked example

```python
import torch as t

# Trailing underscore: same buffer, values doubled.
x = t.tensor([3.0, 1.0, 2.0])
x.mul_(2)
assert x.tolist() == [6.0, 2.0, 4.0]

# No underscore: sort() returns a (values, indices) pair and x is untouched.
result = x.sort()
assert x.tolist() == [6.0, 2.0, 4.0]
assert result.values.tolist() == [2.0, 4.0, 6.0]

# So an in-place sort is "sort, then copy the values back into the buffer".
print("after mul_(2):", x)
x.copy_(x.sort().values)
assert x.tolist() == [2.0, 4.0, 6.0]
print("after copy_(sorted values):", x)
```

Why: the underscore is a contract, not a style. `x.sort()` looks like it
should sort x — in NumPy the same spelling does — and here it quietly does
not.

## Faded practice

### q235
Ascending order, in place — the passed-in object itself must change.

```python starter
import torch as t

def solve(x):
    """Sort x itself (no new tensor), then return it."""
    x._____(x.sort().values)
    return x
```

```python solution
import torch as t

def solve(x):
    """Sort x itself (no new tensor), then return it."""
    x.copy_(x.sort().values)
    return x
```

## Concept: z[:] = expr — writing INTO the buffer

The subtle one: **whole-array in-place assignment.** `z = z[p]` REBINDS the
local name — the caller's array is untouched. `z[:] = z[p]` writes the
values INTO the existing buffer through a full-array slice, so every other
reference to that array sees the change. "Modify the array object passed
in" is this spelling. (Safe with `z[p]` on the right because fancy indexing
copies first.)

## Worked example

```python
import torch as t

# In-place row permutation: write INTO the buffer via z[:].
z = t.arange(6).reshape(3, 2)
alias = z                        # a second reference to the same buffer
p = t.tensor([2, 0, 1])
z[:] = z[p]                      # rebinding (z = z[p]) would NOT affect alias
assert alias.tolist() == [[4, 5], [0, 1], [2, 3]]
print("alias sees the permutation too:")
print(alias)
```

Why: the `alias` variable is the proof of what "in place" means — both
names watch the same memory, so only the `z[:] =` spelling changes what
`alias` sees. This distinction is the entire point of the drill family.

## Faded practice

### q138
Reorder rows by permutation p, in place.

```python starter
import torch as t

def solve(z, p):
    """Row i becomes old row p[i] — INSIDE z's own buffer."""
    z_____ = z[p]
    return z
```

```python solution
import torch as t

def solve(z, p):
    """Row i becomes old row p[i] — INSIDE z's own buffer."""
    z[:] = z[p]
    return z
```

## Concept: chaining underscores for zero allocations

Because each underscore method returns the buffer it just wrote, they chain:
`b.add_(a)` computes a+b and stores it in b — zero new tensors. A run of these
(`a.div_(2)`, `a.neg_()`, …) evaluates a multi-step formula entirely inside
the input buffers.

Two disciplines make such code correct: track *what each buffer now holds*
(after `b.add_(a)`, the NAME b no longer means the original b), and order by
DATAFLOW — a value must be captured before the buffer holding its ingredient
is overwritten.

## Worked example

```python
import torch as t

# ((a + b) * (-a / 2)) in place only. Track buffer contents per step:
a = t.tensor([1.0, 2.0])
b = t.tensor([3.0, 4.0])
b.add_(a)                        # b now holds a + b
a.div_(2)                        # a now holds a / 2
a.neg_()                         # a now holds -a/2
a.mul_(b)                        # a now holds (a+b) * (-a/2)
assert a.tolist() == [-2.0, -6.0]
print("b holds a+b:", b)
print("a holds the final product:", a)
```

Why: order matters — b must absorb (a+b) BEFORE a is halved, since the
addition needs the original a. Reordering the same four calls breaks the
result; dataflow, not formula layout, dictates sequence.

## Faded practice

### q59
((a + b) * (-a / 2)) with ZERO new allocations — every step lands in a's or
b's buffer.

```python starter
import torch as t

def solve(a, b):
    """Compute ((a + b) * (-a / 2)) using only the two given buffers."""
    b.add_(_____)
    a.div_(2)
    a.neg_()
    a.mul_(b)
    return a
```

```python solution
import torch as t

def solve(a, b):
    """Compute ((a + b) * (-a / 2)) using only the two given buffers."""
    b.add_(a)
    a.div_(2)
    a.neg_()
    a.mul_(b)
    return a
```

## Misconceptions

- **"`z = z[p]` modifies z in place."** — It rebinds the NAME; the original
  buffer (and every other reference to it) is unchanged. In-place is
  `z[:] = z[p]` — assignment through the full slice.
- **"`x.sort()` sorts x."** — It does in NumPy; in PyTorch it returns a
  (values, indices) pair and leaves x untouched. There is no `sort_`, so an
  in-place sort is `x.copy_(x.sort().values)`.
- **"The underscore is a naming convention."** — It's a hard contract: the
  result is written into that exact buffer. Aliasing consequences included —
  `b.add_(a)` destroys the old b for all readers. Powerful, deliberate, and
  exactly what no-allocation drills demand.
