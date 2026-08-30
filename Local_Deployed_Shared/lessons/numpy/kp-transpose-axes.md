---
kc: numpy.transpose-axes
title: Transpose — swapping which axis is which
supporting: []
new_syntax: [Tensor.T]
faded: [610, 552, 611, 558, 669, 670]
guided: []
independent: [612, 613, 671, 672, 673, 674]
integrated: [614, 615, 675]
---

## Concept: transpose turns the rows into the columns


You already know a 2-D tensor as a grid: `t.tensor([[1, 2, 3], [4, 5, 6]])` has
two rows and three columns, and its shape says so — `(2, 3)`.

**Transposing** that grid means turning it on its side. Row 0 of the result is
column 0 of the original, row 1 of the result is column 1, and so on. Three
columns going in means three rows coming out, so the shape `(2, 3)` comes back
as `(3, 2)`. Nothing is added, dropped or rounded — the same six numbers are
being written down in a different arrangement.

The spelling is `a.T`, and it is worth saying out loud that **`.T` has no
parentheses**. It is an attribute, like `.shape` and `.ndim`, not a method like
`.tolist()`. Writing `a.T()` is an error, and writing `a.T` where you meant
`a.T.tolist()` hands back a tensor when you asked for a list.

```python
import torch as t

a = t.tensor([[1, 2, 3], [4, 5, 6]])
print(a)
print("shape:", tuple(a.shape))

print(a.T)
print("shape after transposing:", tuple(a.T.shape))
```

Read those two grids against each other. The first row of `a` is `1 2 3`, and
those three numbers came out as the first *column* of `a.T`. That is the whole
operation.

## Worked example


The problem below asks for the transpose's shape. Do it once here, on a tensor
whose two axis lengths are different — which is the only way to see that the
shape really did reverse.

```python
import torch as t

a = t.tensor([[1, 2], [3, 4], [5, 6]])
print("three rows of two:", tuple(a.shape))
print("becomes two rows of three:", tuple(a.T.shape))
```

`tuple(...)` is there because `a.shape` is a `torch.Size`, which prints like a
tuple but is not one; a checker comparing against `(2, 3)` wants the plain
tuple. And notice the shape reversed without you telling PyTorch anything about
lengths — it read both off the tensor.

```python
square = t.tensor([[1, 2], [3, 4]])
print("a square tensor keeps its shape:", tuple(square.T.shape))
print("...but not its contents:", square.T.tolist())
```

That second line is the case worth remembering. A square tensor transposes to
the same *shape*, so shape alone cannot tell you whether a transpose happened.

## Faded practice


### q610
The transpose's shape, as a plain tuple.

```python starter
import torch as t

def solve(rows):
    """Return the transpose's shape as a plain tuple."""
    a = t.tensor(rows)
    return tuple(a.T.shape)
```

```python solution
import torch as t

def solve(rows):
    """Return the transpose's shape as a plain tuple."""
    a = t.tensor(rows)
    return tuple(a.T.shape)
```

### q552
Both shapes at once: the tensor's own, then the transpose's.

```python starter
import torch as t

def solve(rows):
    """Return (a's shape, a.T's shape), both plain tuples."""
    a = t.tensor(rows)
    return (tuple(a.shape), tuple(a.T.shape))
```

```python solution
import torch as t

def solve(rows):
    """Return (a's shape, a.T's shape), both plain tuples."""
    a = t.tensor(rows)
    return (tuple(a.shape), tuple(a.T.shape))
```

### q669
Rows before and rows after — the second is the old column count.

```python starter
import torch as t

def solve(rows):
    """Return (how many rows a has, how many rows a.T has)."""
    a = t.tensor(rows)
    return (len(a), len(a._____))
```

```python solution
import torch as t

def solve(rows):
    """Return (how many rows a has, how many rows a.T has)."""
    a = t.tensor(rows)
    return (len(a), len(a.T))
```

## Concept: reading the transposed numbers back out


The shape tells you the arrangement. To see the numbers themselves, read them
out the same way you read any tensor — with `.tolist()`, which you met on the
previous concept.

```python
import torch as t

a = t.tensor([[1, 2, 3], [4, 5, 6]])
print("a          :", a.tolist())
print("a.T        :", a.T.tolist())
print("a.T row 0  :", a.T.tolist()[0], "  <- column 0 of a")
```

`[1, 4]` is the answer to "what was the first number of each row?" — and that
is what a column *is*. Every row of the transpose answers that question for one
column of the original.

Transposing twice puts every axis back where it started, so `a.T.T` is `a`
again — same shape, same numbers:

```python
back = a.T.T
print("shape is back:", tuple(back.shape) == tuple(a.shape))
print("values are back:", t.equal(back, a))
```

## Worked example


Pulling one row out of a transpose is two steps, and it is worth keeping them
separate the first time: transpose, then read.

```python
import torch as t

a = t.tensor([[10, 20], [30, 40], [50, 60]])

view = a.T                 # step 1: the tensor, turned on its side
rows = view.tolist()       # step 2: the same numbers as plain Python lists

print("every row's first number:", rows[0])
print("every row's second number:", rows[1])
assert rows[0] == [10, 30, 50]
```

## Faded practice


### q611
The transpose's first row — which is the first number of every original row.

```python starter
import torch as t

def solve(rows):
    """Return the transpose's first row as a plain list."""
    a = t.tensor(rows)
    return a.T.tolist()[0]
```

```python solution
import torch as t

def solve(rows):
    """Return the transpose's first row as a plain list."""
    a = t.tensor(rows)
    return a.T.tolist()[0]
```

### q558
The whole transpose, as a plain nested Python list.

```python starter
import torch as t

def solve(rows):
    """Return a.T's contents as a plain nested list."""
    a = t.tensor(rows)
    return a.T.tolist()
```

```python solution
import torch as t

def solve(rows):
    """Return a.T's contents as a plain nested list."""
    a = t.tensor(rows)
    return a.T.tolist()
```

### q670
The last row of the transpose — the last column of the original.

```python starter
import torch as t

def solve(rows):
    """Return the transpose's LAST row as a plain list."""
    a = t.tensor(rows)
    return a._____.tolist()[-1]
```

```python solution
import torch as t

def solve(rows):
    """Return the transpose's LAST row as a plain list."""
    a = t.tensor(rows)
    return a.T.tolist()[-1]
```

## Solo practice

### q612
Transpose twice. Does it come back, and what shape is it then?

### q613
The transpose described two ways: its contents and its shape.

### q671
Is it its own transpose?

### q672
The second row of the transpose, and its length.

### q673
One number, two addresses.

Transposing swaps the two indices of every entry. Whatever sits at row i,
column j of `a` sits at row j, column i of `a.T`:

```python worked
import torch as t

a = t.tensor([[10, 20, 30], [40, 50, 60]])
print("a   at row 0, col 2:", a.tolist()[0][2])
print("a.T at row 2, col 0:", a.T.tolist()[2][0])
assert a.tolist()[0][2] == a.T.tolist()[2][0]
```

### q674
Once, then twice.

## Integrated practice

### q614
Both shapes, the transposed contents, and how many axes there are.

### q615
The element type, the transposed contents, and whether the count survived.

### q675
Everything this page knows about one transpose.

## Misconceptions


- **"`.T` is a method, so it needs parentheses."** — It does not. `a.T` is an
  attribute like `a.shape`; `a.T()` raises. The habit to build is that anything
  *describing* a tensor tends to be an attribute, and anything *doing work* on
  one tends to be a call.
- **"Transposing a square tensor changes nothing."** — It changes the numbers,
  it just cannot change the shape. `[[1, 2], [3, 4]]` transposes to
  `[[1, 3], [2, 4]]` while the shape stays `(2, 2)`, so a shape check will not
  notice a transpose you did or did not mean to do.
- **"`a.T` gives me a list of columns."** — It gives you a *tensor*. It reads
  like a list of the original's columns once you call `.tolist()` on it, but
  until then it is a tensor and behaves like one.
- **"Transposing rearranges the numbers, so it must be slow on a big tensor."**
  — It does not touch them at all. What that costs, and how to tell, is the
  next concept.
