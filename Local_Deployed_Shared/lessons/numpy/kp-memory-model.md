---
kc: numpy.memory-model
title: Memory model — views, reinterpreting bytes, bit unpacking
supporting: [numpy.dtype-astype, numpy.slicing-views, numpy.broadcasting-rules]
new_syntax: []
faded: [123, 204]
guided: []
independent: [113]
---

## Concept: one buffer, one interpretation — view() and storage

You've met views twice (slices; reshapes). This KP makes the underlying model
explicit. A tensor = **one memory buffer (its storage) + an interpretation**
(dtype, shape, strides). Different tensors can interpret the SAME buffer
differently:

- **`z.view(new_dtype)`** — reinterpret the raw bytes under another dtype,
  no conversion. A float32 tensor viewed as int32 shows the IEEE bit patterns
  (garbage as numbers) — which is exactly what makes it the tool for
  *in-place type conversion*: make the view, then ASSIGN the values through
  it (`y = z.view(t.int32); y[:] = z`). The assignment converts (truncating
  float→int like `.to()`), the buffer never moves, and the original float
  tensor is now unusable — one buffer, one live meaning. Contrast the pair:
  **`.to(dtype)` = new buffer, same values; `view(dtype)` = same buffer, new
  interpretation.**
- Torch **overloads `view`**: given a dtype it reinterprets, given a shape
  (`z.view(2, 3)`) it re-shapes. Same method, two jobs — read the argument,
  not the name.
- **`x.untyped_storage().data_ptr()`** — the address of the buffer itself.
  Two tensors share memory exactly when these match, which is how you *prove*
  a view is a view (the graders for in-place drills check exactly this).
  Compare STORAGE pointers, not `x.data_ptr()`: that one includes the view's
  own offset, so `z[1:]` and `z` disagree despite sharing every byte.

These tools are sharp: views break the "tensors are independent" assumption
on purpose. Reach for them when a task says *in place*, *same memory*,
*no copy* — and prove sharing with the storage pointer when it matters.

## Worked example

```python
import torch as t

z = t.tensor([1.5, 99.75, 0.25], dtype=t.float32)

# view = same 12 bytes, reinterpreted. As numbers it's nonsense (bit
# patterns), which is fine — we only want the container.
y = z.view(t.int32)

# Assigning THROUGH the view converts values (truncation) into the buffer.
y[:] = z
assert y.tolist() == [1, 99, 0]
# Same storage => y is a view of z, not a copy of it.
assert y.untyped_storage().data_ptr() == z.untyped_storage().data_ptr()

# .to(dtype) for contrast: correct values immediately, but a NEW buffer.
src = t.tensor([1.5, 2.5], dtype=t.float32)
w = src.to(t.int32)
assert w.untyped_storage().data_ptr() != src.untyped_storage().data_ptr()
```

Why: the two-step in-place conversion (view, then assign through it) is
subtle enough to narrate — the VIEW changes interpretation without touching
bytes; the ASSIGNMENT converts values into those bytes. Either step alone is
wrong. The storage-pointer checks turn "I think this is in place" into a
verified fact.

## Faded practice

### q123
Float32 → int32 in the SAME memory buffer.

```python starter
import torch as t

def solve(z):
    """Return an int32 array sharing z's buffer, holding z's truncated values."""
    y = z._____(t.int32)
    y[:] = z
    return y
```

```python solution
import torch as t

def solve(z):
    """Return an int32 array sharing z's buffer, holding z's truncated values."""
    y = z.view(t.int32)
    y[:] = z
    return y
```

## Concept: bits are data — broadcast AND-masks

Integers are bit patterns too: `v & (1 << k)` tests bit k of every element.
Broadcast a **column of values** against a **row of bit masks**
(`2 ** t.arange(8)`) and you get the (n, 8) matrix of every (element, bit)
test in one expression; `!= 0` booleanizes, `astype(int)` makes 0/1. That
construction is LSB-first — MSB-first output is that matrix column-flipped
(`.flip(1)`; torch rejects a negative slice step, so `[:, ::-1]` is an error
here, not a shortcut). Torch has no packaged `unpackbits` — the broadcast
AND-mask IS the spelling.

(Rare but drilled: subclassing `t.Tensor` — build with
`t.as_tensor(values).as_subclass(cls)` inside `__new__`, and carry attributes
across operations in `__torch_function__`, which is torch's answer to numpy's
`__array_finalize__`. Watch the names: `Tensor.name` already exists and is
read-only, so store the value in `_name` and expose it as a property. See
independent practice.)

## Worked example

```python
import torch as t

# Bits: test each of 8 bit positions against every element at once.
v = t.tensor([1, 128])
bits = ((v.reshape(-1, 1) & (2 ** t.arange(8))) != 0).to(t.int64)
msb_first = bits.flip(1)
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
import torch as t

def solve(v):
    """(len(v), 8) matrix of bits, MSB first."""
    bits = ((v.reshape(-1, 1) & (2 ** t.arange(8))) != 0).to(t.int64)
    return bits[:, _____]
```

```python solution
import torch as t

def solve(v):
    """(len(v), 8) matrix of bits, MSB first."""
    bits = ((v.reshape(-1, 1) & (2 ** t.arange(8))) != 0).to(t.int64)
    return bits.flip(1)
```

## Independent practice

From the drill bank: q113 (a Tensor SUBCLASS carrying a `name` attribute —
`__new__` via `.as_subclass(cls)`, attribute forwarding in
`__torch_function__`; follow the two-method skeleton).

## Misconceptions

- **"view converts values like `.to()`."** — view REINTERPRETS bytes: a
  float32 1.5 viewed as int32 is 1069547520 (its bit pattern). Conversion
  happens only when you assign values through the view. Same buffer + new
  meaning ≠ same values.
- **"`view` means reshape."** — It means both. `z.view(2, 3)` reshapes;
  `z.view(t.int32)` reinterprets. The argument decides.
- **"After `y = z.view(...); y[:] = z`, z still works normally."** — z's
  bytes now hold int patterns; reading z as float32 gives garbage. In-place
  reinterpretation sacrifices the old array — that's the deal.
- **"Bit extraction is string formatting."** — `bin()` + padding is a Python
  loop in costume. The vectorized form is a broadcast AND-mask; it returns
  numbers ready for math, not strings.
