---
kc: numpy.memory-model
title: Memory model — views, reinterpreting bytes, bit unpacking
supporting: [numpy.dtype-astype, numpy.slicing-views, numpy.broadcasting-rules]
new_syntax: []
faded: [123, 204]
guided: []
independent: [113]
---

## Concept: one buffer, one interpretation — view() and .base

You've met views twice (slices; field access). This KP makes the underlying
model explicit. An array = **one memory buffer + an interpretation** (dtype,
shape, strides). Different arrays can interpret the SAME buffer differently:

- **`z.view(new_dtype)`** — reinterpret the raw bytes under another dtype,
  no conversion. A float32 array viewed as int32 shows the IEEE bit patterns
  (garbage as numbers) — which is exactly what makes it the tool for
  *in-place type conversion*: make the view, then ASSIGN the values through
  it (`y = z.view(np.int32); y[:] = z`). The assignment converts (truncating
  float→int like astype), the buffer never moves, and the original float
  array is now unusable — one buffer, one live meaning. Contrast the pair:
  **astype = new buffer, same values; view = same buffer, new
  interpretation.**
- **`x.base`** — who owns the buffer. `None` for an owner; the parent for a
  view. `arr.base is z` is how you *prove* two arrays share memory (graders
  for in-place drills check exactly this).

These tools are sharp: views break the "arrays are independent" assumption
on purpose. Reach for them when a task says *in place*, *same memory*,
*no copy* — and prove sharing with `.base` when it matters.

## Worked example

```python
import numpy as np

z = np.array([1.5, 99.75, 0.25], dtype=np.float32)

# view = same 12 bytes, reinterpreted. As numbers it's nonsense (bit
# patterns), which is fine — we only want the container.
y = z.view(np.int32)

# Assigning THROUGH the view converts values (truncation) into the buffer.
y[:] = z
assert y.tolist() == [1, 99, 0]
assert y.base is not None            # y is a view — same memory as z

# astype for contrast: correct values immediately, but a NEW buffer.
w = np.array([1.5, 2.5], dtype=np.float32).astype(np.int32)
assert w.base is None                # w owns fresh memory
```

Why: the two-step in-place conversion (view, then assign through it) is
subtle enough to narrate — the VIEW changes interpretation without touching
bytes; the ASSIGNMENT converts values into those bytes. Either step alone is
wrong. `.base` checks turn "I think this is in place" into a verified fact.

## Faded practice

### q123
Float32 → int32 in the SAME memory buffer.

```python starter
import numpy as np

def solve(z):
    """Return an int32 array sharing z's buffer, holding z's truncated values."""
    y = z._____(np.int32)
    y[:] = z
    return y
```

```python solution
import numpy as np

def solve(z):
    """Return an int32 array sharing z's buffer, holding z's truncated values."""
    y = z.view(np.int32)
    y[:] = z
    return y
```

## Concept: bits are data — broadcast AND-masks

Integers are bit patterns too: `v & (1 << k)` tests bit k of every element.
Broadcast a **column of values** against a **row of bit masks**
(`2 ** np.arange(8)`) and you get the (n, 8) matrix of every (element, bit)
test in one expression; `!= 0` booleanizes, `astype(int)` makes 0/1. That
construction is LSB-first — MSB-first output is that matrix column-flipped
(`[:, ::-1]`). (`np.unpackbits(v.astype(np.uint8)[:, None], axis=1)` is the
packaged spelling.)

(Rare but drilled: subclassing `np.ndarray` — create with
`np.asarray(values).view(cls)` inside `__new__`, forward attributes in
`__array_finalize__`. See independent practice.)

## Worked example

```python
import numpy as np

# Bits: test each of 8 bit positions against every element at once.
v = np.array([1, 128])
bits = ((v.reshape(-1, 1) & (2 ** np.arange(8))) != 0).astype(int)
msb_first = bits[:, ::-1]
assert msb_first[0].tolist() == [0, 0, 0, 0, 0, 0, 0, 1]   # 1
assert msb_first[1].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]   # 128
```

Why: the bit matrix is broadcasting doing systems work — column of values ×
row of masks = every (element, bit) test in one expression. LSB-first falls
out naturally; MSB-first is one flip.

## Faded practice

### q204
Each unsigned integer becomes its 8 bits, MOST significant first.

```python starter
import numpy as np

def solve(v):
    """(len(v), 8) matrix of bits, MSB first."""
    bits = ((v.reshape(-1, 1) & (2 ** np.arange(8))) != 0).astype(int)
    return bits[:, _____]
```

```python solution
import numpy as np

def solve(v):
    """(len(v), 8) matrix of bits, MSB first."""
    bits = ((v.reshape(-1, 1) & (2 ** np.arange(8))) != 0).astype(int)
    return bits[:, ::-1]
```

## Independent practice

From the drill bank: q113 (an ndarray SUBCLASS carrying a `name` attribute —
`__new__` via `.view(cls)`, attribute forwarding in `__array_finalize__`;
follow the two-method skeleton).

## Misconceptions

- **"view converts values like astype."** — view REINTERPRETS bytes: a
  float32 1.5 viewed as int32 is 1069547520 (its bit pattern). Conversion
  happens only when you assign values through the view. Same buffer + new
  meaning ≠ same values.
- **"After `y = z.view(...); y[:] = z`, z still works normally."** — z's
  bytes now hold int patterns; reading z as float32 gives garbage. In-place
  reinterpretation sacrifices the old array — that's the deal.
- **"Bit extraction is string formatting."** — `bin()` + padding is a Python
  loop in costume. The vectorized form is a broadcast AND-mask (or
  np.unpackbits); it returns numbers ready for math, not strings.
