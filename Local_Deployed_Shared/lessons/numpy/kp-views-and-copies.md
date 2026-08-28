---
kc: numpy.views-and-copies
title: Views and copies — the same numbers, read a different way
supporting: []
new_syntax: [Tensor.contiguous, Tensor.data_ptr, Tensor.is_contiguous]
faded: [616, 617, 618, 536]
guided: []
independent: [553, 523]
integrated: [535, 561, 565]
---

## Concept: one strip of numbers, and a note saying how to read it


Your computer does not store a grid. It stores **one long strip of numbers, one
after another**. For `[[1, 2, 3], [4, 5, 6]]` the strip in memory is literally

    1 2 3 4 5 6

and that is all of it. There is nothing two-dimensional anywhere in memory.

The grid comes from a small **note** attached to that strip, saying how to read
it: *"treat this as 2 rows by 3 columns; to step one place right, move forward
1 number; to step one place down, move forward 3."* Follow the note and the
grid comes back. Shape is the first half of that note.

Two words get used constantly from here on, and both of them are about the
strip rather than the grid:

- **"the block"** (or "the buffer", or "the storage") is the strip itself — the
  actual run of numbers in memory. Two tensors *share a block* when both of
  their notes point at the same strip.
- **"in reading order"** — the technical word is **contiguous** — means the
  numbers sit on the strip in exactly the order you would read them off the
  grid, left to right and top to bottom, with no hopping. A tensor you have
  just built is always in reading order: `t.tensor` writes the strip by reading
  the grid.

PyTorch will answer both questions for you. `a.data_ptr()` is *"what address
does my strip start at?"* — a big integer that is meaningless on its own and
tells you everything when you compare two of them. `a.is_contiguous()` is
*"are my numbers in reading order?"*

```python
import torch as t

a = t.tensor([[1, 2, 3], [4, 5, 6]])

print("a starts at:", type(a.data_ptr()).__name__, "- an address, not a value")
print("a is in reading order:", a.is_contiguous())
print("a compared with itself:", a.data_ptr() == a.data_ptr())
```

Now the point of all this. Transposing does **not** touch the strip. It writes
a *new note*: *"treat this as 3 rows by 2 columns; to step one place right,
move forward 3; to step one place down, move forward 1."* Same strip, new note.

```python
view = a.T   # same cell as above — `a` is still the 2x3 tensor

print("a.T starts at the same address:", view.data_ptr() == a.data_ptr())
print("a.T is in reading order:", view.is_contiguous())
print("a.T reads as:", view.tolist())
```

Both answers together are the whole idea. `a.T` **shares a's block** — it is
reading the very same `1 2 3 4 5 6`. And it is **not in reading order**,
because reading the transposed grid gives `1 4 2 5 3 6`, which is not the order
the strip is written in. It only produces that order by hopping.

A tensor that borrows someone else's block like this is called a **view**.

## Worked example


The two questions are asked with two different calls, and it is worth running
them side by side once on a tensor whose transpose is genuinely out of order.

```python
import torch as t

a = t.tensor([[1, 2], [3, 4], [5, 6]])
view = a.T

print("strip order of a  :", a.tolist(), "-> reads 1 2 3 4 5 6")
print("grid order of a.T :", view.tolist(), "-> reads 1 3 5 2 4 6")

print("same strip?      ", view.data_ptr() == a.data_ptr())
print("in reading order?", view.is_contiguous())
```

The freshly built tensor is always in reading order, so the interesting answer
is always the second one:

```python
print("a itself:", a.is_contiguous())
```

## Faded practice


### q616
Two bools: is `a` in reading order, and is its transpose?

```python starter
import torch as t

def solve(rows):
    """Return (is a in reading order?, is a.T?)."""
    a = t.tensor(rows)
    return (a.is_contiguous(), a.T.is_contiguous())
```

```python solution
import torch as t

def solve(rows):
    """Return (is a in reading order?, is a.T?)."""
    a = t.tensor(rows)
    return (a.is_contiguous(), a.T.is_contiguous())
```

## Concept: sharing a strip and being in order are two different questions


It is easy to hear "shares memory" and "is contiguous" as one idea. They are
not, and a transpose is the standard case where they disagree: it shares the
strip *and* is out of order.

Every combination is reachable, so neither answer implies the other:

```python
import torch as t

a = t.tensor([[1, 2, 3], [4, 5, 6]])

print("a.T      shares:", a.T.data_ptr() == a.data_ptr(),
      " in order:", a.T.is_contiguous())          # shares, out of order

fresh = t.tensor([[1, 2, 3], [4, 5, 6]])
print("a rebuild shares:", fresh.data_ptr() == a.data_ptr(),
      " in order:", fresh.is_contiguous())        # own strip, in order
```

There is one shape where the transpose *is* already in reading order, and it
catches people out: when the tensor has only one row, or only one column. A
`(1, 4)` tensor transposes to `(4, 1)`, and reading a single column top to
bottom walks the strip straight through — no hopping, so nothing is out of
order.

```python
strip_like = t.tensor([[1, 2, 3, 4]])
print("one row, transposed:", strip_like.T.tolist())
print("still in reading order:", strip_like.T.is_contiguous())
```

🔴 **`data_ptr()` is a STARTING address, not the identity of the strip.** For a
transpose the two are the same question, because a transpose starts on the same
number `a` does. A tensor that starts PART-WAY along the strip — a slice — is
still reading `a`'s strip while answering a different starting address:

```python
later = a[1:]
print("a[1:] starts where a does:", later.data_ptr() == a.data_ptr())
print("...but it is the same strip:",
      later.untyped_storage().data_ptr() == a.untyped_storage().data_ptr())
```

So `x.data_ptr() == y.data_ptr()` reads as "do these two start at the same
place", and that is exactly the question every drill on this page asks. When
you want "is this the same strip at all", regardless of where each tensor
begins, the question to ask is `x.untyped_storage().data_ptr()`.

## Worked example


Both questions, on the same tensor, in the order a checker would ask them.

```python
import torch as t

a = t.tensor([[7, 8, 9], [1, 2, 3]])
view = a.T

shares = view.data_ptr() == a.data_ptr()
in_order = view.is_contiguous()

print("(shares, in_order) =", (shares, in_order))
assert shares is True
assert in_order is False
```

Getting these two the wrong way round is the single most common mistake on this
concept, because both answers are bools and the tuple still looks plausible.

## Faded practice


### q617
Two bools about the same transpose: does it start where `a` does, and is it in reading order?

```python starter
import torch as t

def solve(rows):
    """Return (does a.T start where a does?, is a.T in reading order?)."""
    a = t.tensor(rows)
    view = a.T
    return (view.data_ptr() == a.data_ptr(), view.is_contiguous())
```

```python solution
import torch as t

def solve(rows):
    """Return (does a.T start where a does?, is a.T in reading order?)."""
    a = t.tensor(rows)
    view = a.T
    return (view.data_ptr() == a.data_ptr(), view.is_contiguous())
```

## Concept: asking for a strip of your own — .contiguous()


Sometimes hopping is not good enough and you want the numbers actually written
out in the order you read them. **`.contiguous()`** is how you ask. The name is
the technical one; what it does is plain enough:

> *"Give me a tensor holding these numbers in reading order."*

For `a.T`, whose grid reads `1 4 2 5 3 6` while its strip says `1 2 3 4 5 6`,
the only way to do that is to write `1 4 2 5 3 6` somewhere else in memory. A
new strip means a new starting address — the result does **not** share `a`'s
block any more. That is the copy.

```python
import torch as t

a = t.tensor([[1, 2, 3], [4, 5, 6]])
view = a.T
packed = view.contiguous()

print("same numbers as the view:", t.equal(packed, view))
print("in reading order now    :", packed.is_contiguous())
print("still a's block         :", packed.data_ptr() == a.data_ptr())
```

🔴 **`.contiguous()` does not always copy, and this is the part that surprises
people.** If the tensor is already in reading order there is nothing to do, so
it hands the *same tensor* straight back — same strip, same address. So "a
packed copy never shares the original's block" is false; it is only true when
the thing you packed was actually out of order.

```python
print("a is already in order:", a.is_contiguous())
print("so a.contiguous() is a itself:", a.contiguous().data_ptr() == a.data_ptr())

one_row = t.tensor([[1, 2, 3, 4]])
print("a one-row transpose is in order too:", one_row.T.is_contiguous())
print("so ITS packed copy shares:",
      one_row.T.contiguous().data_ptr() == one_row.data_ptr())
```

That is why a drill on this can answer `(True, False)` for a 2×3 input and
`(True, True)` for a `[[9]]` or `[[1, 2]]` input without contradicting itself.

## Worked example


The three-step move — transpose, pack, compare — written out once.

```python
import torch as t

a = t.tensor([[1, 2], [3, 4], [5, 6]])

view = a.T                  # borrows a's strip, out of order
packed = view.contiguous()  # writes its own strip, in order

print("view   shares:", view.data_ptr() == a.data_ptr(),
      " in order:", view.is_contiguous())
print("packed shares:", packed.data_ptr() == a.data_ptr(),
      " in order:", packed.is_contiguous())
print("same values either way:", t.equal(packed, view))
```

## Faded practice


### q618
Pack two different tensors and ask where each copy starts: `a`'s, then its transpose's.

```python starter
import torch as t

def solve(rows):
    """Return (does a's packed copy start where a does?, does a.T's?)."""
    a = t.tensor(rows)
    return (a.contiguous().data_ptr() == a.data_ptr(),
            a.T.contiguous().data_ptr() == a.data_ptr())
```

```python solution
import torch as t

def solve(rows):
    """Return (does a's packed copy start where a does?, does a.T's?)."""
    a = t.tensor(rows)
    return (a.contiguous().data_ptr() == a.data_ptr(),
            a.T.contiguous().data_ptr() == a.data_ptr())
```

### q536
Two bools: is the transpose in reading order, and is its packed copy?

```python starter
import torch as t

def solve(rows):
    """Return (is a.T in reading order?, is its packed copy?)."""
    a = t.tensor(rows)
    view = a.T
    return (view.is_contiguous(), view.contiguous().is_contiguous())
```

```python solution
import torch as t

def solve(rows):
    """Return (is a.T in reading order?, is its packed copy?)."""
    a = t.tensor(rows)
    view = a.T
    return (view.is_contiguous(), view.contiguous().is_contiguous())
```

## Solo practice

### q553
Three tensors, one question each: `a`, its transpose, and the packed copy — which are in reading order?

### q523
The transpose and its packed copy: which one reads `a`'s block, and what do the numbers look like?

## Integrated practice

### q535
Two bools about the same pair: does the transpose read `a`'s block, and does the packed copy?

### q561
Shape, transposed shape, element type, and whether the transpose shares the block.

### q565
Four bools about a transpose and its packed copy: order, order, values, and whose memory.

## Misconceptions


- **"Shares memory" and "is contiguous" are the same question.** — They are
  independent. A transpose shares the block *and* is out of reading order,
  which is exactly the case every drill here is built on.
- **"A packed copy never shares the original's block."** — Only when the thing
  being packed was out of order. `.contiguous()` on a tensor that is already in
  reading order returns that tensor unchanged, address and all, which is why a
  single-row or single-column input answers `True` where a 2×3 answers `False`.
- **"`data_ptr()` tells me something about the values."** — It is an address.
  On its own it means nothing; it is only ever useful compared against another
  tensor's.
- **"`data_ptr()` equal means same block, unequal means different block."** —
  The second half is wrong. It is the address a tensor STARTS at, so a slice
  that begins part-way along someone else's strip answers a different address
  while sharing every byte. `x.untyped_storage().data_ptr()` is the question
  that ignores where each tensor begins.
- **"Transposing is expensive because it moves the numbers."** — It moves
  nothing and costs almost nothing. `.contiguous()` is the call that pays, and
  it pays because it writes a whole new strip.
- **"A view is a copy that happens to look the same."** — The opposite. A view
  is a second note on someone else's strip; write through one and the other
  sees it.
