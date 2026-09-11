---
kc: python.control-flow
title: Decisions and repeated work
new_syntax: ['syntax.if', 'syntax.else', 'syntax.ternary', 'syntax.is', 'syntax.for', 'syntax.aug-assign', 'syntax.comprehension', 'syntax.comprehension-filter']
concepts: [branch, inline-choice, accumulator, comprehension]
supporting: ['python.defining-functions', 'python.lists-and-tuples', 'python.indexing', 'numpy.boolean-masking']
previews: []
faded: [1281, 1282, 1283, 1284]
guided: []
independent: [1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293]
integrated: [1294, 1295, 1296]
---

## Concept: A branch chooses which code runs

An `if` statement runs its indented block only when its condition is true. An `else` block handles every remaining case. The general procedure is: write down the cases the input can be in, decide what the function should do in each, and only then write the branch. The reason to state the cases first is that the branch you forget is the one the program silently gets wrong — an empty list, an option that was never given, a batch with one example. Naming each case in prose turns "it crashed on empty input" into a line of code you can point at.

A condition is any expression that is true or false: a comparison such as `len(xs)==0`, or a test such as `value>0`. When the true block returns, the code after the whole `if` is the else case, so an explicit `else` is optional — but it is clearer when both branches assign the same name.

```python
xs=[]
if len(xs)==0:
    result=7
else:
    result=xs[0]
print(result)
# Hidden checks
assert result==7
```

Here the two cases are "empty" and "has a first item", and `result` is set in both, so nothing after the branch has to wonder whether the name exists.

## Worked example

We want the sign of a number as a word. The cases are negative, zero and positive, so the branch needs three arms: `if`, an `elif` for the middle case, and `else` for the rest. Predict the word before running.

```python
n=-4
if n<0:
    word="negative"
elif n==0:
    word="zero"
else:
    word="positive"
print(word)
# Hidden checks
assert word=="negative"
```

The `elif` is tested only when the first condition failed, so `n==0` never has to repeat `not n<0`. Now change the input and check that the middle arm is reachable — a case you never exercise is a case you never tested.

```python
n=0
if n<0:
    word="negative"
elif n==0:
    word="zero"
else:
    word="positive"
print(word)
# Hidden checks
assert word=="zero"
```

## Faded practice

### q1281
Return the first value of xs, or fallback when xs is empty. xs: list of integers, possibly empty; fallback: integer.

```python starter
def solve(xs,fallback):
    pass
```

```python solution
def solve(xs,fallback):
    if len(xs)==0:
        return fallback
    return xs[0]
```

## Concept: An inline choice selects one value

When the two branches differ only in the value they produce, Python has an expression form: `a if condition else b`. It evaluates to `a` when the condition holds and to `b` otherwise, and it can sit anywhere a value can — on the right of `=`, inside a `return`, inside a list. Use it when each arm is a single value; use the statement form when an arm needs more than one line, because a long inline choice hides the cases instead of naming them.

A related decision is how to represent an option that was not given. Python uses `None` for absence and tests it with `is None`. The reason to test with `is` rather than `==` is that `None` is a single object: `x is None` asks "is this the absent marker?", while `x==0` would also be false for an absent value and true for a real zero. Zero is a value; `None` is the lack of one.

```python
offset=None
y=3 if offset is None else 3+offset
print(y)
# Hidden checks
assert y==3
```

## Worked example

A layer may or may not have a bias. We compute the output as the product plus the bias when a bias was given, and just the product when it was not. Predict `out` before running.

```python
product=10
bias=None
out=product if bias is None else product+bias
print(out)
# Hidden checks
assert out==10
```

Now give a bias of zero. The output is unchanged — but for a different reason: the bias exists and adds nothing. A test written as `bias==0` would have treated the two cases the same; `is None` keeps them apart.

```python
bias=0
out=product if bias is None else product+bias
print(out, bias is None)
# Hidden checks
assert out==10 and bias is not None
```

## Faded practice

### q1282
Return xs when fallback is zero, otherwise return a one-item list containing fallback. xs: list of integers, possibly empty; fallback: integer.

```python starter
def solve(xs,fallback):
    pass
```

```python solution
def solve(xs,fallback):
    return xs if fallback==0 else [fallback]
```

## Concept: A loop carries a small piece of state

A `for` loop visits each item of a list in order. To compute something about the whole list, keep an accumulator: a variable that records what has been learned so far, updated once per item. The procedure is to decide, before the loop, what the accumulator means and what its value is when nothing has been seen yet; then write the one update that keeps the meaning true after each item. If the meaning holds before and after every step, it holds at the end — that is the whole argument for why the loop is correct.

The starting value is the identity of the update: `0` for a sum, `1` for a product, an empty list for collecting. `total += value` is shorthand for `total = total + value`: read the current value, combine, write it back. Choosing the wrong start — `1` for a sum, say — is the classic accumulator bug, and it shows first on the empty list, where the start value is the answer.

```python
xs=[2,-1,4]
total=0
for value in xs:
    total += value
print(total)
# Hidden checks
assert total==5
```

## Worked example

We count how many items of a list are even. The accumulator `count` means "even items seen so far", so it starts at zero and grows by one only inside the branch. Predict the count before running.

```python
xs=[4,7,10,3]
count=0
for value in xs:
    if value%2==0:
        count += 1
print(count)
# Hidden checks
assert count==2
```

On an empty list the loop body never runs, and the start value is returned untouched — zero even items, which is right. That is the empty case checking itself.

```python
count=0
for value in []:
    if value%2==0:
        count += 1
print(count)
# Hidden checks
assert count==0
```

## Faded practice

### q1283
Return the sum of xs, using a loop. Empty lists total zero. xs: list of integers, possibly empty.

```python starter
def solve(xs):
    pass
```

```python solution
def solve(xs):
    total=0
    for value in xs:
        total += value
    return total
```

## Concept: A comprehension builds a list from independent outputs

When every output depends only on its own input item, write a list comprehension: `[expression for item in xs]`. It produces a new list of the same length, one result per item, in order. A trailing `if` — `[expression for item in xs if condition]` — keeps only the items that pass the test, so the result can be shorter. The rule for choosing between a comprehension and a loop is whether the outputs depend on each other: a running total needs an accumulator, but squaring every item does not, and the comprehension says so in one line.

The filter runs before the expression, so the expression only ever sees items that passed. That ordering is what makes `[1/x for x in xs if x!=0]` safe: the zero is dropped before it is divided by.

```python
xs=[2,-1,4]
positive_squares=[value*value for value in xs if value>0]
print(positive_squares)
# Hidden checks
assert positive_squares==[4,16]
```

## Worked example

We convert a list of lengths in metres to centimetres, then keep only the ones under a threshold. The first comprehension has no filter, so its output has the same length as the input.

```python
metres=[0.5,2.0,1.25]
cm=[100*m for m in metres]
print(cm)
# Hidden checks
assert cm==[50.0,200.0,125.0]
```

Adding the filter drops items; the survivors keep their original order. Predict which lengths remain before running.

```python
short=[100*m for m in metres if m<1.5]
print(short)
# Hidden checks
assert short==[50.0,125.0]
```

## Faded practice

### q1284
Return squares of positive values in xs, in original order. xs: list of integers, possibly empty.

```python starter
def solve(xs):
    pass
```

```python solution
def solve(xs):
    return [value*value for value in xs if value>0]
```

## Solo practice

### q1285
Return the last item when xs is nonempty, otherwise fallback. xs: list of integers, possibly empty; fallback: integer.

### q1286
Return the sum of all positive items. xs: list of integers, possibly empty.

### q1287
Return a list replacing negative items with fallback, preserving other items and order. xs: list of integers, possibly empty; fallback: integer.

### q1288
Return the product of all items. The empty product is one. xs: list of integers, possibly empty.

### q1289
Return the first positive item, or fallback if none exists. xs: list of integers, possibly empty; fallback: integer.

### q1290
Return the last positive item, or fallback if none exists. xs: list of integers, possibly empty; fallback: integer.

### q1291
Return items strictly greater than the average of xs, in original order. Empty input returns an empty list. xs: list of integers, possibly empty.

### q1292
Return the largest item, or fallback for an empty list. Do not assume items are positive. xs: list of integers, possibly empty; fallback: integer.

### q1293
Return the length of the longest uninterrupted run of positive items. xs: list of integers, possibly empty.

## Integrated practice

### q1294
Return the value nearest to fallback; ties choose the earliest item. Empty input returns fallback. xs: list of integers, possibly empty; fallback: integer.

### q1295
Return how many times xs changes sign between consecutive nonzero values. Zeros are ignored, not separators. xs: list of integers, possibly empty.

### q1296
Return the maximum prefix sum, including the empty prefix with sum zero. xs: list of integers, possibly empty.

## Misconceptions

- **Zero is absence.** `0` is a value and `None` is the lack of one; test for absence with `is None`, not `==0`.
- **A loop with no matching item returns nothing.** It returns the accumulator's start value — which is why the start must be the identity of the update (`0` for a sum, `1` for a product).
- **A comprehension can carry state.** It cannot; each output sees only its own item. A running total or "changes since the previous item" needs a loop.
- **`elif` re-tests the earlier conditions.** It runs only when every earlier arm failed, so its condition can assume they did.
