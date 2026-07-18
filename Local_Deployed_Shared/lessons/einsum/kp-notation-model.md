---
kc: einsum.notation-model
title: Reading an einsum spec string
supporting: [numpy.axis-reductions, numpy.dot-matmul-patterns, numpy.reshape-flatten]
new_syntax: [einsum-spec-string]
faded: [244]
guided: [271]
independent: [285]
---

## Concept

`np.einsum` is one function that expresses transposes, reductions, dots,
matrix products, and most of deep learning's tensor plumbing — by letting you
**name the axes**. The call is

> `np.einsum('SPEC', arr1, arr2, ...)`

where the spec string labels each input's axes with letters and declares the
output's axes after `->`. Reading a spec is a fixed four-step ritual:

1. **Left of `->`**: one letter per axis of each input, inputs separated by
   commas. `'ij'` on a matrix means "call axis 0 `i`, axis 1 `j`". The
   letters are arbitrary names — only their PATTERN matters.
2. **Right of `->`**: which axes the output has, in what order.
3. **A letter that appears in the output** is kept (and its position sets
   the output layout).
4. **A letter that appears in inputs but NOT in the output is SUMMED
   over.** This is the entire computational content of the notation.

With just rules 3–4 you can already read two big families:

- **Pure axis permutation** (every letter kept, order changed):
  `'ij->ji'` is the transpose — output axis 0 is what the input called `j`.
  `'bchw->bhwc'` moves channels last: same data, relabeled positions —
  einsum as a self-documenting alternative to `np.transpose` with axis
  numbers.
- (Preview of the next KP: drop a letter instead — `'ij->i'` — and you've
  summed an axis.)

Why bother, when transpose/sum/dot all exist? Because the spec string *is
the documentation*: `'bchw->bhwc'` states the memory layout change in
domain terms, and complex multi-input contractions (coming in this lesson)
have no readable spelling without it. Deep-learning code is full of einsum
for exactly this reason.

## Worked example

Task: transpose a matrix, and relayout a channels-first image batch to
channels-last — both as pure relabelings.

```python
import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]])

# 'ij->ji': axis names i (rows), j (cols); output wants (j, i).
# Both letters survive -> nothing is summed; it's a pure permutation.
t = np.einsum('ij->ji', a)
assert t.tolist() == [[1, 4], [2, 5], [3, 6]]
assert np.array_equal(t, a.T)               # same thing, named vs method

# 'bchw->bhwc': 4 axes named batch, channel, height, width;
# output keeps all four, channel moved last.
imgs = np.arange(24).reshape(2, 3, 2, 2)     # (b=2, c=3, h=2, w=2)
last = np.einsum('bchw->bhwc', imgs)
assert last.shape == (2, 2, 2, 3)
# One value check: batch 0, channel 1, position (h=1, w=0)...
assert imgs[0, 1, 1, 0] == last[0, 1, 0, 1]  # ...is now indexed (b,h,w,c)

# The ritual, applied to a spec you haven't seen: 'xy->yx' on a 2-D array
# is ALSO the transpose — letters are names, only the pattern matters.
assert np.array_equal(np.einsum('xy->yx', a), a.T)
```

Why each step:

1. For `'ij->ji'`, say the ritual aloud: "inputs: i then j; output: j then
   i; every letter kept → permutation only." Thirty seconds of narration per
   spec is what builds fluency.
2. The single-value check on the 4-D case (`imgs[0,1,1,0] == last[0,1,0,1]`)
   is how to verify ANY relayout: pick one element, track where its indices
   moved. Cheaper and more convincing than eyeballing whole arrays.
3. The `'xy->yx'` variant kills the superstition that `i` and `j` are magic:
   spec letters are bound variables, like loop variables in
   `for i ... for j ...`.

## Faded practice

### q244
The transpose, written as an axis relabeling.

```python starter
import numpy as np

def solve(a):
    """Transpose via einsum: name the axes, emit them swapped."""
    return np.einsum('_____', a)
```

```python solution
import numpy as np

def solve(a):
    """Transpose via einsum: name the axes, emit them swapped."""
    return np.einsum('ij->ji', a)
```

## Guided practice

### q271
1. (b, c, h, w) → (b, h, w, c): four axes in, four out — which family is
   this, permutation or reduction?
2. Name the input axes something meaningful ('bchw') and write the output in
   the required order.
3. All letters kept on both sides — nothing summed, values untouched.

## Independent practice

From the drill bank: q285 (swap the middle two axes of a 4-D tensor — write
the spec, then verify one element's journey like the worked example).

## Misconceptions

- **"einsum's letters have fixed meanings (i = rows, b = batch)."** — They
  are arbitrary bound names; `'ij->ji'` and `'qz->zq'` are the same program.
  Choose letters that document YOUR tensor (b, c, h, w for images) — that's
  convention for humans, not semantics for NumPy.
- **"einsum always multiplies/sums something."** — A spec that keeps every
  letter is a pure transpose/relayout. The summing behavior is triggered
  only by DROPPING letters (next KP) or repeating them across inputs.
- **"The output side is optional decoration."** — Everything hangs on it:
  which axes survive, their order, and (by omission) what gets summed.
  Implicit mode (no `->`) exists but reorders alphabetically — in this
  course, always write the arrow.
