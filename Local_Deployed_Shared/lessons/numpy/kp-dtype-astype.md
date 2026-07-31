---
kc: numpy.dtype-astype
title: Dtypes, .to(), and memory size
supporting: [numpy.ndarray-model]
new_syntax: []
faded: [230, 215, 51]
guided: [87]
independent: [19]
---

## Concept: .to() — converting an existing tensor

Every tensor has exactly one **dtype** — the type shared by all of its
elements. It determines what the values can be: `int64` can't hold 0.5;
`float32` holds it with less precision than `float64`; `bool` holds only
`True`/`False`. Mixed inputs get promoted to the common type (ints + one
float → all float), silently.

To convert an existing tensor: **`x.to(new_dtype)`** (this is PyTorch's
`astype`). It returns a *new* tensor when the dtype actually changes — the
original is untouched; there is no in-place dtype change. Converting
float→int **truncates toward zero** rather than rounding: `1.9 → 1`,
`-1.9 → -1`.

```python
import torch as t

x = t.tensor([1, 2, 3])
y = x.to(t.float32)
print(x, x.dtype)
print(y, y.dtype)
assert x.dtype == t.int64          # the original never changed
```

Truncation is the part that surprises people — it is not rounding:

```python
print(t.tensor([1.9, -1.9, 0.5]).to(t.int64))
assert t.tensor([1.9]).to(t.int64).item() == 1
assert t.tensor([-1.9]).to(t.int64).item() == -1
```

One wrinkle worth knowing: if the dtype you ask for is the one it already
has, `.to()` hands back the *same* tensor rather than a copy. It promises a
tensor of that dtype, not a fresh buffer.

```python
same = x.to(t.int64)
print("same object?", same is x)
assert same is x
```

## Worked example

Make a float32 copy of an integer tensor:

```python
import torch as t

x = t.tensor([1, 2, 3])
assert x.dtype == t.int64           # default integer dtype

# .to() returns a NEW tensor with converted values; x is untouched.
y = x.to(t.float32)
assert y.dtype == t.float32
assert x.dtype == t.int64           # original unchanged — .to() copies
print("x", x, x.dtype)
print("y", y, y.dtype)
```

Why: checking `x.dtype` first tells you what conversion is actually needed —
don't convert blind. And dtype is a property of the whole memory block, so
changing it means building a new block.

## Faded practice

### q230
Float32 copy of an integer tensor, original left unmodified.

```python starter
import torch as t

def solve(x):
    """Return a float32 copy of integer tensor x."""
    return x._____(t.float32)
```

```python solution
import torch as t

def solve(x):
    """Return a float32 copy of integer tensor x."""
    return x.to(t.float32)
```

## Concept: dtypes at creation, and dtype names as strings

Rather than build-then-convert, request the dtype at creation: every
constructor accepts `dtype=`, e.g. `t.arange(n, dtype=t.int32)` — one
step, no copy.

```python
import torch as t

built = t.arange(4, dtype=t.int32)
print(built, built.dtype)
```

Here PyTorch is stricter than NumPy. NumPy accepts the *string* `'float32'`
anywhere a dtype is wanted; PyTorch does not — `t.arange(3, dtype='float32')`
raises a `TypeError`. Watch it happen:

```python
try:
    t.arange(3, dtype='float32')
except TypeError as err:
    print("TypeError:", err)
```

When the dtype arrives as **data** (from a config, a file header, a function
argument), you have to turn the name into the dtype object first, and
`getattr(t, name)` does exactly that.

```python
for name in ('int32', 'float32', 'float64'):
    z = t.arange(3, dtype=getattr(t, name))
    print(f"{name:8} -> {z} {z.dtype}")
assert t.arange(3, dtype=getattr(t, 'float32')).dtype == t.float32
```

## Worked example

Build 0..2 already in float32 — dtype named by a string:

```python
import torch as t

name = 'float32'

# PyTorch wants the dtype OBJECT, so look it up from the name.
z = t.arange(3, dtype=getattr(t, name))
assert z.dtype == t.float32
print(repr(name), "->", getattr(t, name), "->", z)
```

Why: when a function receives `dtype_str` as an argument, one `getattr` turns
it into the real dtype — no lookup table from strings to torch objects, and no
`if/elif` chain over dtype names.

## Faded practice

### q215
The integers 0..n-1 stored with a dtype named by a string.

```python starter
import torch as t

def solve(n, dtype_str):
    """0..n-1 with the dtype named by dtype_str (e.g. 'int32')."""
    return t.arange(n, dtype=_____(t, dtype_str))
```

```python solution
import torch as t

def solve(n, dtype_str):
    """0..n-1 with the dtype named by dtype_str (e.g. 'int32')."""
    return t.arange(n, dtype=getattr(t, dtype_str))
```

## Concept: dtype is a memory choice — element_size and numel

The number in a dtype's name is bits: `int32` = 4 bytes per element,
`float64` = 8. Total buffer size is `elements × bytes-per-element` —
**`x.numel()`** elements at **`x.element_size()`** bytes each. Dtype choices
are memory choices: halving precision halves the buffer, which is the whole
reason models train in float32 (or bfloat16) rather than float64.

```python
import torch as t

grid = t.zeros((10, 10))
for dtype in (t.float64, t.float32, t.int16, t.bool):
    z = grid.to(dtype)
    print(f"{str(dtype):15} {z.numel()} x {z.element_size()} = "
          f"{z.numel() * z.element_size()} bytes")
```

Same 100 numbers, an 8× spread in what they cost:

```python
assert grid.to(t.float64).element_size() == 2 * grid.to(t.float32).element_size()
assert grid.to(t.float32).numel() * grid.to(t.float32).element_size() == 400
print("float64 is exactly", grid.to(t.float64).element_size(), "bytes/element,",
      "float32", grid.to(t.float32).element_size())
print("100 float32 elements =",
      grid.to(t.float32).numel() * grid.to(t.float32).element_size(), "bytes")
```

## Worked example

```python
import torch as t

x = t.tensor([1, 2, 3])             # int64: 8 bytes each
assert x.numel() * x.element_size() == 3 * 8

# float32 elements are 4 bytes, so the converted copy is half the size.
y = x.to(t.float32)
assert y.element_size() == 4
assert y.numel() * y.element_size() == 3 * 4
print(x.dtype, x.numel() * x.element_size(), "bytes")
print(y.dtype, y.numel() * y.element_size(), "bytes")
```

Why: `numel × element_size` makes the cost concrete — a 10×10 float32
tensor is exactly 400 bytes, where the float64 NumPy equivalent is 800.

## Faded practice

### q51
A tensor's buffer size, reported as the string "&lt;n&gt; bytes".

```python starter
import torch as t

def solve(z):
    """Return z's data-buffer size as e.g. '24 bytes'."""
    return f"{z.numel() * z._____()} bytes"
```

```python solution
import torch as t

def solve(z):
    """Return z's data-buffer size as e.g. '24 bytes'."""
    return f"{z.numel() * z.element_size()} bytes"
```

## Guided practice

### q87
1. Torch has a histogram counter that takes the bin count and the range
   directly — no bucketing by hand.
2. It returns FLOAT counts. The drill wants integers, so the last step is
   a dtype conversion.
3. `t.histc(z, bins=bins, min=0.0, max=1.0).to(t.int64)` — this is the
   dtype lesson in miniature: the numbers were already right, only their
   type was wrong.

## Independent practice

From the drill bank: q19 (split a complex tensor into real and imaginary
parts — `z.real` / `z.imag`; `complex64` values carry two floats).

## Misconceptions

- **"`.to()` changes the tensor in place."** — It returns a converted copy;
  the original keeps its dtype and values. If you meant to keep it, assign it:
  `x = x.to(...)`.
- **"Float→int conversion rounds."** — It truncates toward zero: `1.9 → 1`,
  `-1.9 → -1`. If you want rounding, round first: `x.round().to(t.int64)`.
- **"`dtype='float32'` works, like in NumPy."** — PyTorch rejects the string
  with a `TypeError`. Pass `t.float32`, or `getattr(t, name)` when the name
  arrives as data.
- **"dtype names are just labels."** — The number is the bit width, which sets
  both the representable range/precision and the memory per element.
  `numel × element_size` — a 10×10 float32 tensor is exactly 400 bytes.
