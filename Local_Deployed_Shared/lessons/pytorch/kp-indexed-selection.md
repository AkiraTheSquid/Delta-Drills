---
kc: tensor.indexed-selection
title: Selecting one entry per row
new_syntax: ['Tensor.gather', 'Tensor.abs']
concepts: [paired-indices, gather]
supporting: ['numpy.slicing-views', 'numpy.ranges', 'numpy.axis-reductions', 'numpy.broadcasting-rules']
previews: []
faded: [1020, 1022, 1019, 1021]
guided: []
independent: [1023, 1024, 1025, 1026, 1027, 1028, 1077, 1078, 1079]
integrated: [1029, 1030, 1031]
---

## Concept: Paired index tensors pick one entry per row

An integer tensor used as an index is a list of requests: `prices[items]` returns the entries at those positions, in request order, repeats included. Indexing is not sorting and not de-duplication; it answers exactly the questions asked. With two index tensors, one per axis, the requests are paired up position by position: `x[rows, cols]` returns `x[rows[0], cols[0]]`, then `x[rows[1], cols[1]]`, and so on, so the result has the shape of the index tensors, not of `x`.

The common need is "the value at column `ids[i]` of row `i`" — the score a classifier gave the true class, the chosen action's reward. The row index is then just `0, 1, 2, …`, which is `t.arange(b)`, and the column index is `ids` itself: `x[t.arange(b), ids]` has shape `(b,)`. The reason to build the row index explicitly is that `x[:, ids]` would mean something else — every row at every requested column, shape `(b, b)` — because a slice on the row axis is not paired with the column requests, it is crossed with them.

```python
import torch as t
x=t.tensor([[10.,20.,30.],[40.,50.,60.]])
ids=t.tensor([2,0])
chosen=x[t.arange(2),ids]
print(chosen)
# Hidden checks
assert chosen.tolist()==[30.,40.]
```

## Worked example

A batch of three examples has logits for four classes, and `y` holds the true class of each example. We want each example's score for its own true class. The pairs are `(0, y[0])`, `(1, y[1])`, `(2, y[2])`, so the row index is `arange(3)`.

```python
import torch as t
logits=t.tensor([[1.,5.,2.,0.],[3.,1.,4.,1.],[0.,0.,2.,9.]])
y=t.tensor([1,2,3])
true_scores=logits[t.arange(3),y]
print(true_scores)
# Hidden checks
assert true_scores.tolist()==[5.,4.,9.]
```

Compare with the crossed form. `logits[:, y]` takes columns 1, 2 and 3 of every row: a `(3, 3)` matrix whose diagonal is the paired answer and whose off-diagonal entries are scores for other examples' classes.

```python
crossed=logits[:,y]
print(crossed.shape, crossed)
# Hidden checks
assert crossed.shape==(3,3) and crossed[1,1].item()==4.
```

## Faded practice

### q1020
Return requested values as a vector, shape (b,). x: float (b,c); ids: one valid column index per row (b,).

```python starter
import torch as t

def solve(x,ids):
    pass
```

```python solution
import torch as t

def solve(x,ids):
    return x[t.arange(x.shape[0]),ids]
```

### q1022
Return requested values minus each row’s mean, shape (b,). x: float (b,c); ids: one valid column index per row (b,).

```python starter
import torch as t

def solve(x,ids):
    pass
```

```python solution
import torch as t

def solve(x,ids):
    return x[t.arange(len(ids)),ids]-x.mean(dim=1)
```

## Concept: Gather takes an index tensor with the same rank as the input

`x.gather(dim, index)` is the same one-entry-per-row selection written as one call. Along `dim`, `index` says which position to read; every other axis is walked in lockstep. So for a `(b, c)` matrix, `x.gather(1, index)` with `index` of shape `(b, 1)` reads `x[i, index[i, 0]]` for every row `i` and returns a `(b, 1)` tensor — the output always has the shape of `index`. When the contract asks for `(b,)`, either squeeze that last axis or use the paired-index form.

The reason `index` must have the same rank as `x` is that gather is defined per output position: for each coordinate of `index` it reads one value, and it needs a coordinate on every axis of `x` to know where. Turning a `(b,)` vector of column ids into the required `(b, 1)` is `ids[:, None]`. Gather earns its keep when the selection has to broadcast back against the rows — subtract each row's chosen value from the whole row — because a `(b, 1)` result already lines up with `(b, c)`.

A companion when the selected values are compared by size is `x.abs()`, which drops the sign of every entry: a chosen logit of `-3` and one of `3` are equally far from zero.

```python
import torch as t
x=t.tensor([[2.,-7.,5.],[8.,4.,-9.]])
cols=t.tensor([[1],[2]])
picked=x.gather(1,cols)
print(picked, picked.abs())
# Hidden checks
assert picked.tolist()==[[-7.],[-9.]] and picked.abs().tolist()==[[7.],[9.]]
```

## Worked example

Each row is a set of readings and `ids` names the sensor to report per row. We gather with a `(b, 1)` index built from `ids`, then use the `(b, 1)` shape directly to centre every row on its reported sensor.

```python
import torch as t
readings=t.tensor([[3.,-1.,6.],[0.,5.,-2.]])
ids=t.tensor([2,1])
picked=readings.gather(1,ids[:,None])
print(picked.shape, picked)
# Hidden checks
assert picked.shape==(2,1) and picked.tolist()==[[6.],[5.]]
```

Because `picked` is `(2, 1)`, subtracting it from the `(2, 3)` readings broadcasts one value across each row. The reported sensor reads zero in every row, and `abs` then gives each other sensor's distance from it.

```python
centred=readings-picked
print(centred.abs())
# Hidden checks
assert centred.abs().tolist()==[[3.,7.,0.],[5.,0.,7.]]
```

## Faded practice

### q1019
Return requested value per row, shape (b,1). x: float (b,c); ids: one valid column index per row (b,).

```python starter
import torch as t

def solve(x,ids):
    pass
```

```python solution
import torch as t

def solve(x,ids):
    return x.gather(1,ids[:,None])
```

### q1021
Return the sum of requested values, as a scalar tensor. x: float (b,c); ids: one valid column index per row (b,).

```python starter
import torch as t

def solve(x,ids):
    pass
```

```python solution
import torch as t

def solve(x,ids):
    return x.gather(1,ids[:,None]).sum()
```

## Solo practice

### q1023
Return requested value per row, shape (b,). x: float (b,c); ids: one valid column index per row (b,).

### q1024
Return the gap from each requested value to its row’s maximum, shape (b,). x: float (b,c); ids: one valid column index per row (b,).

### q1025
Return whether the requested value is a row maximum, shape (b,); ties count as maxima. x: float (b,c); ids: one valid column index per row (b,).

### q1026
Return requested entries with their row order reversed, shape (b,). x: float (b,c); ids: one valid column index per row (b,).

### q1027
Return each row after subtracting its requested value, shape (b,c). x: float (b,c); ids: one valid column index per row (b,).

### q1028
Return each requested column across all rows, preserving request order, shape (b,b). x: float (b,c); ids: one valid column index per row (b,).

### q1077
Return the largest requested value across examples, as a scalar tensor. x: float (b,c); ids: one valid column index per row (b,).

### q1078
Return how often each column index was requested, shape (c,). x: float (b,c); ids: one valid column index per row (b,).

### q1079
Return the absolute difference between each requested value and the requested value of the first row, shape (b,). x: float (b,c); ids: one valid column index per row (b,).

## Integrated practice

### q1029
Return the sum of absolute requested values, as a scalar tensor. x: float (b,c); ids: one valid column index per row (b,).

### q1030
Return the requested value divided by the sum of absolute values in its row, shape (b,). Rows have positive absolute sum. x: float (b,c); ids: one valid column index per row (b,).

### q1031
Return each row’s zero-based rank of its requested value: count entries strictly greater, shape (b,). Ties share a rank. x: float (b,c); ids: one valid column index per row (b,).

## Misconceptions

- **`x[:, ids]` picks one entry per row.** A slice is crossed with the requests, not paired: the result is `(b, b)`. Pairing needs an explicit row index.
- **Gather's output has the shape of `x`.** It has the shape of `index`; a `(b, 1)` index gives a `(b, 1)` result even when `x` is `(b, c)`.
- **Repeated requests are collapsed.** Indexing returns one value per request, duplicates included; a basket with the same item twice is charged twice.
- **`abs` is only for negative numbers.** It is the distance from zero for every entry, which is what "how far" questions need regardless of sign.
