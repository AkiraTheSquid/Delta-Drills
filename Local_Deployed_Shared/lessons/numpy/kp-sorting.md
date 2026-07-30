---
kc: numpy.sorting
title: Sorting tensors
supporting: [numpy.ndarray-model, numpy.slicing-views]
new_syntax: [torch.sort, torch.sort#descending, torch.sort#dim, torch.argsort, torch.argsort#descending, torch.argsort#dim, torch.topk, Tensor.values, Tensor.indices]
faded: [58, 520, 521]
guided: [516, 517]
independent: [518, 519, 522]
---

## Concept: sort returns a pair, not a tensor

**`t.sort(z)` does not return a sorted tensor.** It returns a *pair* — the
sorted values and the indices that produced them — and forgetting that is the
mistake this KP exists to prevent:

```python no-run
t.sort(z)          # -> torch.return_types.sort(values=..., indices=...)
t.sort(z).values   # the sorted tensor you actually wanted
```

```python
import torch as t

z = t.tensor([0.5, 0.25, 0.75])
print(t.sort(z))
```

That printout is the pair, and it is what a function returns if you forget
`.values`.

You can unpack it either way: `values, indices = t.sort(z)`, or reach for
`.values` / `.indices` by name. The indices half is not a consolation prize —
it is what "sort one thing by another" tasks need, and it is the same thing
`t.argsort(z)` gives you on its own.

```python
values, indices = t.sort(z)
print("values ", values)
print("indices", indices)
assert t.equal(indices, t.argsort(z))
assert t.equal(z[indices], values)      # the indices reconstruct the values
```

Sorting never modifies the input: `t.sort(z)` and the method form `z.sort()`
both leave `z` alone and hand back a new pair. (There is no in-place `sort_`.)

**`descending=True`** sorts largest-first — a real keyword, unlike NumPy, where
you have to sort ascending and reverse afterwards.

```python
print("z is still", z)
print("descending", t.sort(z, descending=True).values)
assert z.tolist() == [0.5, 0.25, 0.75]
```

## Worked example

Task: produce a sorted copy of a vector, confirm the original is intact, then
get the same values descending.

```python
import torch as t

z = t.tensor([0.5, 0.25, 0.75])

# sort returns a PAIR — take .values for the sorted tensor.
result = t.sort(z)
assert result.values.tolist() == [0.25, 0.5, 0.75]
assert result.indices.tolist() == [1, 0, 2]     # where each value came from

# The input keeps its original order (the grader often checks this).
assert z.tolist() == [0.5, 0.25, 0.75]

# Descending is a keyword here — no reverse step needed.
desc = t.sort(z, descending=True).values
assert desc.tolist() == [0.75, 0.5, 0.25]

# Unpacking works too, and reads well when you want both halves.
values, indices = t.sort(z)
assert values.tolist() == [0.25, 0.5, 0.75]
```

Why each step:

1. Taking `.values` is the habit to build. Returning `t.sort(z)` straight from
   a function hands the caller a pair, and the failure looks like a type error
   far from the line that caused it.
2. The indices are the bridge to the order-statistics KP: they say *where*
   each sorted value came from, which is how you carry a second tensor along.
3. `descending=True` is one of the places PyTorch is friendlier than NumPy —
   worth knowing so you don't write the reverse-slice workaround (which
   wouldn't work here anyway, since negative slice steps are rejected).

## Faded practice

### q58
Sorted copy, smallest to largest, input left unmodified.

```python starter
import torch as t

def solve(z):
    """Return a NEW tensor with z's values in ascending order."""
    return t.sort(z)._____
```

```python solution
import torch as t

def solve(z):
    """Return a NEW tensor with z's values in ascending order."""
    return t.sort(z).values
```

## Concept: argsort — the positions, and reordering by them

`t.argsort(z)` gives you the indices half on its own: `order[0]` is the position
of the smallest element, `order[1]` the next, and so on. On a single tensor that
is just a slower route to `t.sort(z).values` — you would still have to index with
it.

Its real job is **carrying a second tensor along**. When two tensors are parallel
— names and scores, boxes and confidences, tokens and logits — sorting one of
them independently destroys the correspondence. So you rank once, get the order,
and index *every* tensor with that same order. They stay lined up because they
all moved the same way.

Indexing a tensor with an index tensor is the "fancy indexing" you have already
met: `names[order]` builds a new tensor by reading `names` at each position in
`order`, in that order.

```python
import torch as t

ids = t.tensor([10, 11, 12, 13])
scores = t.tensor([0.4, 0.9, 0.1, 0.7])

order = t.argsort(scores, descending=True)
print("order  ", order)
print("ids    ", ids[order])
print("scores ", scores[order])
```

Both tensors moved by the SAME order, so row-by-row they still describe the
same items. Sorting them separately is the bug this pattern prevents:

```python
broken = t.sort(ids, descending=True).values
print("independently sorted ids:", broken, "— no longer paired with anything")
assert t.equal(scores[order], t.sort(scores, descending=True).values)
```

## Worked example

```python
import torch as t

names = t.tensor([10, 20, 30])
scores = t.tensor([0.5, 0.25, 0.75])

# The positions that WOULD sort scores ascending — not the values.
order = t.argsort(scores)
assert order.tolist() == [1, 0, 2]

# Index both tensors with the SAME order and they stay in correspondence.
assert scores[order].tolist() == [0.25, 0.5, 0.75]
assert names[order].tolist() == [20, 10, 30]

# argsort takes the same direction keyword sort does.
best_first = t.argsort(scores, descending=True)
assert best_first.tolist() == [2, 0, 1]
assert names[best_first].tolist() == [30, 10, 20]
```

Read the last two lines: name 30 comes first because score 0.75 is the highest,
and nothing about `names` was sorted. That is the whole pattern — rank one
tensor, index all of them.

## Faded practice

### q520
Rank by score, then move both tensors with the same order.

```python starter
import torch as t

def solve(names, scores):
    """Return (names, scores) as lists, both ordered highest score first."""
    order = t.argsort(scores, _____=True)
    return (names[_____].tolist(), scores[_____].tolist())
```

```python solution
import torch as t

def solve(names, scores):
    """Return (names, scores) as lists, both ordered highest score first."""
    order = t.argsort(scores, descending=True)
    return (names[order].tolist(), scores[order].tolist())
```

## Concept: picking an axis, and taking only the top k

On a tensor with more than one axis, **`dim=`** says which axis to sort along,
and every lane along that axis is sorted independently. `t.sort(m, dim=1)` sorts
*within* each row — which means it scrambles each row's contents and destroys any
correspondence between columns. That is usually not what you want from "sort the
matrix"; reordering whole rows is the argsort-plus-indexing pattern from the last
segment, applied to axis 0.

```python
import torch as t

m = t.tensor([[3.0, 1.0, 2.0],
              [9.0, 7.0, 8.0]])
print("dim=1 (within each row)")
print(t.sort(m, dim=1).values)
print("dim=0 (down each column)")
print(t.sort(m, dim=0).values)
```

**`t.topk(z, k)`** answers a narrower question: the k largest values, already
ordered largest first, plus their positions. Sorting the whole tensor to slice
off k of them does strictly more work — and on a long vector, a great deal more.
Like `sort`, it hands back a `(values, indices)` pair.

```python
v = t.tensor([5.0, 1.0, 9.0, 3.0, 7.0])
top = t.topk(v, 2)
print(top)
assert top.values.tolist() == [9.0, 7.0]
assert t.equal(top.values, t.sort(v, descending=True).values[:2])
```

## Worked example

```python
import torch as t

m = t.tensor([[3.0, 1.0, 2.0], [9.0, 7.0, 8.0]])

# dim=1 sorts WITHIN each row, independently of the other rows.
assert t.sort(m, dim=1).values.tolist() == [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]

# dim=0 does the same down the columns.
assert t.sort(m, dim=0).values.tolist() == [[3.0, 1.0, 2.0], [9.0, 7.0, 8.0]]

# argsort takes the same dim= keyword, and answers with positions not values.
assert t.argsort(m, dim=1).tolist() == [[1, 2, 0], [1, 2, 0]]

# ...and the same descending= keyword you met a segment ago.
z = t.tensor([0.5, 0.25, 0.75, 0.125])
assert t.argsort(z, descending=True).tolist() == [2, 0, 1, 3]

# topk: the k largest, largest first — a pair again, so take .values.
top = t.topk(z, 2)
assert top.values.tolist() == [0.75, 0.5]
assert top.indices.tolist() == [2, 0]
```

The second assertion is the one to sit with: sorting down the columns left this
matrix unchanged, because every column was already ascending. Sorting along an
axis tells you nothing about the other axis.

## Faded practice

### q521
Per-row rankings, largest first — an axis and a direction at the same time.

```python starter
import torch as t

def solve(x):
    """Return each row's column indices ordered by that row's values, largest first."""
    return t.argsort(x, _____=1, _____=True)
```

```python solution
import torch as t

def solve(x):
    """Return each row's column indices ordered by that row's values, largest first."""
    return t.argsort(x, dim=1, descending=True)
```

## Guided practice

### q516
1. A NEW tensor, so the input has to survive untouched.
2. Sorting returns both values and the positions they came from; you want
   the values half, and there is a keyword for the direction.
3. `t.sort(z, descending=True).values`.

### q517
1. Not the sorted values — where each sorted value WOULD have come from.
2. The result is an index tensor the same length as z, and result[0] is the
   position of the smallest element.
3. `t.argsort(z)`.

## Independent practice

From the drill bank: q518 (sort each row of a matrix independently).
From the drill bank: q519 (the k largest values — sorting everything does more work than asked).
From the drill bank: q522 (sort a matrix's ROWS by one column, keeping each row intact).

## Misconceptions

- **"`t.sort(z)` returns the sorted tensor."** — It returns a
  `(values, indices)` pair. Take `.values`, or unpack both.
- **"`z.sort()` sorts in place, like the NumPy method."** — It does not. There
  is no in-place sort; the input is never modified.
- **"There's no descending option, so I'll reverse the result."** — There is:
  `descending=True`. And the NumPy reversal idiom `[::-1]` would raise here
  anyway.
- **"Sorting a 2-D tensor sorts the rows as units."** — `t.sort(z, dim=1)`
  sorts *within* each row independently, destroying row integrity. Keeping
  rows intact while reordering them is an argsort + indexing pattern (later
  KP).
