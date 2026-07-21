---
kc: numpy.dtype-astype
title: Dtypes, astype, and memory size
supporting: [numpy.ndarray-model]
new_syntax: []
faded: [230, 215, 51]
guided: []
independent: [19]
---

## Concept: astype — converting an existing array

Every array has exactly one **dtype** — the type shared by all of its
elements. It determines what the values can be: `int64` can't hold 0.5;
`float32` holds it with less precision than `float64`; `bool` holds only
`True`/`False`. Mixed inputs get promoted to the common type (ints + one
float → all float), silently.

To convert an existing array: **`x.astype(new_dtype)`**. This always returns
a *new* array (a copy) — the original is untouched; there is no in-place
dtype change. Converting float→int **truncates toward zero** rather than
rounding: `1.9 → 1`, `-1.9 → -1`.

## Worked example

Make a float32 copy of an integer array:

```python
import numpy as np

x = np.array([1, 2, 3])
assert x.dtype == np.int64          # default integer dtype

# astype returns a NEW array with converted values; x is untouched.
y = x.astype(np.float32)
assert y.dtype == np.float32
assert x.dtype == np.int64          # original unchanged — astype copies
```

Why: checking `x.dtype` first tells you what conversion is actually needed —
don't convert blind. And dtype is a property of the whole memory block, so
changing it means building a new block.

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

## Concept: dtypes at creation, and dtype names as strings

Rather than build-then-convert, request the dtype at creation: every
constructor accepts `dtype=`, e.g. `np.arange(n, dtype=np.int32)` — one
step, no copy.

A dtype has three equivalent spellings — the NumPy object `np.float32`, the
string `'float32'`, or `np.dtype('float32')`. Strings are handy when the
dtype arrives as *data* (from a config, a file header, a function argument):
they're accepted anywhere a dtype is.

## Worked example

Build 0..2 already in float32 — dtype named by a string:

```python
import numpy as np

# Constructor + dtype in one step — no astype copy needed.
z = np.arange(3, dtype='float32')
assert z.dtype == np.float32
```

Why: when a function receives `dtype_str` as an argument, pass it straight
through — no lookup table from strings to NumPy objects is needed.

## Faded practice

### q215
The integers 0..n-1 stored with a dtype named by a string.

```python starter
import numpy as np

def solve(n, dtype_str):
    """0..n-1 with the dtype named by dtype_str (e.g. 'int32')."""
    return np.arange(n, _____=dtype_str)
```

```python solution
import numpy as np

def solve(n, dtype_str):
    """0..n-1 with the dtype named by dtype_str (e.g. 'int32')."""
    return np.arange(n, dtype=dtype_str)
```

## Concept: dtype is a memory choice — itemsize and nbytes

The number in a dtype's name is bits: `int32` = 4 bytes per element,
`float64` = 8. Total buffer size is `elements × bytes-per-element`,
available directly as **`arr.nbytes`** (and per-element as `arr.itemsize`).
Dtype choices are memory choices: halving precision halves the buffer.

## Worked example

```python
import numpy as np

x = np.array([1, 2, 3])             # int64: 8 bytes each
assert x.nbytes == 3 * 8

# float32 elements are 4 bytes, so the converted copy is half the size.
y = x.astype(np.float32)
assert y.itemsize == 4
assert y.nbytes == 3 * 4
```

Why: `nbytes = size × itemsize` makes the cost concrete — a 10×10 float64
array is exactly 800 bytes.

## Faded practice

### q51
An array's buffer size, reported as the string "&lt;n&gt; bytes".

```python starter
import numpy as np

def solve(z):
    """Return z's data-buffer size as e.g. '24 bytes'."""
    return f"{z._____} bytes"
```

```python solution
import numpy as np

def solve(z):
    """Return z's data-buffer size as e.g. '24 bytes'."""
    return f"{z.nbytes} bytes"
```

## Independent practice

From the drill bank: q19 (split a complex array into real and imaginary
parts — `np.real(z)` / `np.imag(z)`; `complex128` values carry two floats).

## Misconceptions

- **"astype changes the array in place."** — It returns a converted COPY;
  the original keeps its dtype and values. If you meant to keep it, assign it:
  `x = x.astype(...)`.
- **"Float→int conversion rounds."** — It truncates toward zero: `1.9 → 1`,
  `-1.9 → -1`. If you want rounding, round first: `np.round(x).astype(int)`.
- **"dtype names are just labels."** — The number is the bit width, which sets
  both the representable range/precision and the memory per element.
  `nbytes = size × itemsize` — a 10×10 float64 array is exactly 800 bytes.
