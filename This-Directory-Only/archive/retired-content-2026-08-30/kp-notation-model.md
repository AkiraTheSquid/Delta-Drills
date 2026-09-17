---
kc: einsum.notation-model
title: Reading an einsum spec string
supporting: [numpy.axis-reductions, numpy.dot-matmul-patterns, numpy.reshape-flatten]
new_syntax: [torch.einsum]
concepts: [spec-anatomy, output-order-permutation]
faded: [244, 285]
guided: [271]
independent: [300, 303]
---

## Concept: anatomy of the spec string

`t.einsum` does one thing: it lets you **name the axes** of your tensors, then
describe the result by those names. Every call looks like

> `t.einsum('SPEC', tensor1, tensor2, ...)`

and the whole operation lives in that `'SPEC'` string. The string has exactly
two sides, split by `->`:

    'ik , kj -> ij'
     input1  input2   output

Read each slot:

- **Before `->` — the inputs.** One group per tensor you pass in, groups
  separated by a **comma**. Each group gives **one letter per axis**, in order:
  `ik` means "this tensor is 2-D; call axis 0 `i` and axis 1 `k`." The comma is
  literally "here comes the next tensor." Two groups → two tensors.
- **After `->` — the output.** The axes you want in the result, in the order
  you want them.
- **A letter is just a NAME for an axis** — like a loop variable. `i`, `x`, `q`
  carry no built-in meaning; only the PATTERN of where a letter appears matters.

Where each letter appears decides its fate — the three fates are the whole
language:

1. **Kept** — appears in the output → that axis survives (this lesson).
2. **Summed** — appears in an input but NOT the output → that axis is added up
   and disappears (next lesson).
3. **Multiplied** — the same letter shared across two inputs → those axes are
   paired and multiplied (a couple of lessons on).

Why name axes instead of using `permute`/`sum` with axis numbers?
Because the spec string IS the documentation: `'bchw->bhwc'` says "move channels
last" in plain axis names, where `transpose(x, (0,2,3,1))` makes you count.

## Watch out

- **Letters have no fixed meaning.** `'ij->ij'` and `'xy->xy'` are the same
  program; `i` is not "rows," `b` is not "batch" — those are conventions for
  humans reading your code, not rules PyTorch knows.
- **Always write the `->`.** Implicit mode (no arrow) exists but guesses the
  output by sorting letters alphabetically. In this course, always state the
  output side yourself.

## Worked example

The clearest way to see the grammar is the spec that does NOTHING: name every
axis, then ask for them back in the same order.

```python
import torch as t

a = t.tensor([[1, 2, 3],
              [4, 5, 6]])

# 'ij->ij': input is 2-D (axis 0 = i, axis 1 = j); output lists i then j,
# the same order -> every axis kept, nothing reordered = the tensor unchanged.
same = t.einsum('ij->ij', a)
print(same)
# tensor([[1, 2, 3],
#         [4, 5, 6]])   <- exactly a, handed straight back
```

Why this is the anchor: `'ij->ij'` hands the tensor straight back — `print(same)`
shows the same numbers you started with. That proves the grammar in isolation:
the letters name the two axes, and the output side requests them unchanged.
Every other einsum spec is just a DEVIATION from this do-nothing baseline:
reorder the output letters (transpose), drop one (sum), or share one across two
inputs (multiply). Learn the baseline, and the rest are edits to it.

(The code is preloaded in the editor on the right — press Run and watch the
Output pane print the tensor — PyTorch wraps it in `tensor(...)`, which
is the surest sign you are looking at a tensor and not a list. That's the
fastest way to convince yourself.)

## Faded practice

### q244
Same tensor, but now emit the two axes in the OPPOSITE order — rows become
columns. You saw `'ij->ij'` return the tensor unchanged; change only the output
side so the layout flips (this is the transpose).

```python starter
import torch as t

def solve(a):
    """Transpose via einsum: name the axes, then emit them in swapped order."""
    return t.einsum('_____', a)
```

```python solution
import torch as t

def solve(a):
    """Transpose via einsum: name the axes, then emit them in swapped order."""
    return t.einsum('ij->ji', a)
```

## Concept: reordering the output = permutation

When every input letter also appears in the output — none dropped, none shared
— nothing is summed or multiplied. The ONLY thing the output side can do is set
the ORDER of the axes. That is a pure axis permutation.

- On a 2-D tensor, the only reorder is the transpose: `'ij->ji'`.
- On an N-D tensor, you can move any axis anywhere, and the spec documents the
  move in domain terms. `'bchw->bhwc'` takes a batch of channels-FIRST images
  `(batch, channel, height, width)` and lays them out channels-LAST
  `(batch, height, width, channel)` — the same numbers, re-indexed. The letters
  say exactly which axis went where; `x.permute(0, 2, 3, 1)` makes you
  decode the tuple.

No data changes — a permutation only relabels positions.

## Watch out

- **The output side is not decoration.** It decides which axes survive AND their
  order. Here every letter is kept, so it only reorders — but omit a letter and
  that axis would be SUMMED away (next lesson). Every letter you write, and every
  one you leave out, means something.

## Worked example

```python
import torch as t

imgs = t.arange(24).reshape(2, 3, 2, 2)   # (b=2, c=3, h=2, w=2), channels-first

# 'bchw->bhwc': keep all four axes, move channel to the end.
last = t.einsum('bchw->bhwc', imgs)
print(last.shape)          # (2, 2, 2, 3)  <- channels moved to the end

# Follow ONE element to see it only changed POSITION, not value:
print(imgs[0, 1, 1, 0])    # 5   (at b=0, c=1, h=1, w=0 in the original)
print(last[0, 1, 0, 1])    # 5   (same value, now at b=0, h=1, w=0, c=1)
```

Why: to check a relayout, print the shape and follow a single element — cheaper
and surer than eyeballing whole tensors. The shape shows channels landed last,
and the two prints both read `5`: the value didn't change, it only moved from
position `(b,c,h,w)=(0,1,1,0)` to `(b,h,w,c)=(0,1,0,1)`. That's what "permutation
relabels positions, never values" means, made concrete.

## Faded practice

### q285
A 4-D tensor `(b, h, s, d)`. This time don't move an axis to the end — SWAP the
two MIDDLE axes (`h` and `s`), leaving `b` first and `d` last. Same "reorder the
kept letters" idea as the channels-last move, a different target order.

```python starter
import torch as t

def solve(x):
    """(b, h, s, d) -> (b, s, h, d): swap the two middle axes, ends fixed."""
    return t.einsum('_____', x)
```

```python solution
import torch as t

def solve(x):
    """(b, h, s, d) -> (b, s, h, d): swap the two middle axes, ends fixed."""
    return t.einsum('bhsd->bshd', x)
```

## Guided practice

### q271
1. `(b, c, h, w) -> (b, h, w, c)`: four axes in, four out — permutation or
   reduction?
2. Name the input axes meaningfully (`bchw`) and write the output in the
   required order.
3. Every letter kept on both sides — nothing summed, values untouched:
   `'bchw->bhwc'`.

## Independent practice

From the drill bank: q300 (flatten the two spatial axes, channels
preserved), q303 (the antisymmetric part a - a.T, straight from the
transpose subscript).
