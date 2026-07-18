---
kc: numpy.dtype-astype
title: Dtypes, astype, and memory size
supporting: [numpy.ndarray-model]
new_syntax: []
faded: [230]
guided: [215]
independent: [51, 19]
---

## Concept

Every array has exactly one **dtype** — the type shared by all of its
elements. It determines three things you will care about again and again:

1. **What the values can be.** `int64` can't hold 0.5; `float32` holds it with
   less precision than `float64`; `bool` holds only `True`/`False`.
2. **How much memory each element takes.** The number in the name is bits:
   `int32` = 4 bytes per element, `float64` = 8. Total buffer size is
   `elements × bytes-per-element`, available directly as **`arr.nbytes`**
   (and per-element as `arr.itemsize`).
3. **What happens when types meet.** Mixed inputs get promoted to the common
   type (ints + one float → all float), silently.

The general procedures:

- **Convert an existing array**: **`x.astype(new_dtype)`**. This always
  returns a *new* array (a copy) — the original is untouched. Converting
  float→int **truncates toward zero** rather than rounding.
- **Request a dtype at creation**: every constructor accepts `dtype=`, e.g.
  `np.arange(n, dtype=np.int32)`.
- **Name a dtype**: three equivalent spellings — the NumPy object
  `np.float32`, the string `'float32'`, or `np.dtype('float32')`. Strings are
  handy when the dtype arrives as data (from a config, a file header, a
  function argument).

Special dtypes to recognize: `bool` (comparisons produce these), and
`complex128`, whose values carry two floats — extract the pieces with
`np.real(z)` / `np.imag(z)`.

## Worked example

Task: take an integer array, produce a float32 copy for a model that expects
32-bit inputs, and confirm what that did to memory.

```python
import numpy as np

x = np.array([1, 2, 3])
assert x.dtype == np.int64          # default integer dtype (8 bytes each)
assert x.nbytes == 3 * 8            # 3 elements x 8 bytes

# astype returns a NEW array with converted values; x is untouched.
y = x.astype(np.float32)
assert y.dtype == np.float32
assert y.tolist() == [1.0, 2.0, 3.0]
assert x.dtype == np.int64          # original unchanged — astype copies

# float32 elements are 4 bytes, so the copy is half the size.
assert y.itemsize == 4
assert y.nbytes == 3 * 4

# The dtype could just as well arrive as a string:
z = np.arange(3, dtype=np.dtype('float32'))
assert z.dtype == y.dtype
```

Why each step:

1. Checking `x.dtype` first tells you what conversion is actually needed —
   don't convert blind.
2. `astype` rather than reassigning elements: dtype is a property of the whole
   memory block, so changing it means building a new block. There is no
   in-place dtype change (reinterpreting bytes is a different, advanced
   operation covered in the memory-model KP).
3. `nbytes`/`itemsize` make the cost concrete: dtype choices are memory
   choices. Halving precision halves the buffer.

The truncation gotcha, once:

```python
f = np.array([1.9, -1.9])
assert f.astype(int).tolist() == [1, -1]   # toward zero — NOT rounding
```

## Faded practice

### q230
Float32 copy of an integer array, original left unmodified.

```python starter
import numpy as np

def solve(x):
    """Return a float32 copy of integer array x."""
    return x._____(np.float32)
```

```python solution
import numpy as np

def solve(x):
    """Return a float32 copy of integer array x."""
    return x.astype(np.float32)
```

## Guided practice

### q215
1. You're asked for the integers 0..n-1 stored with a dtype whose NAME arrives
   as a string like `'int32'`.
2. You could build then convert, but every constructor takes `dtype=`
   directly — one step, no copy.
3. A dtype string is accepted anywhere a dtype is: `np.arange(n, dtype=...)`
   works with `'int32'` or `np.dtype(dtype_str)`.

## Independent practice

From the drill bank: q51 (report an array's buffer size as "&lt;n&gt; bytes" —
one attribute does it), q19 (split a complex array into real and imaginary
parts).

## Misconceptions

- **"astype changes the array in place."** — It returns a converted COPY;
  the original keeps its dtype and values. If you meant to keep it, assign it:
  `x = x.astype(...)`.
- **"Float→int conversion rounds."** — It truncates toward zero: `1.9 → 1`,
  `-1.9 → -1`. If you want rounding, round first: `np.round(x).astype(int)`.
- **"dtype names are just labels."** — The number is the bit width, which sets
  both the representable range/precision and the memory per element.
  `nbytes = size × itemsize` — a 10×10 float64 array is exactly 800 bytes.
