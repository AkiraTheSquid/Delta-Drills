---
kc: python.defining-functions
title: Writing your own function — def and return
supporting: [python.calling-functions]
new_syntax: [syntax.def, syntax.default-argument, syntax.docstring, syntax.return]
faded: [598, 599]
guided: []
independent: [600, 601]
integrated: [602, 603]
---

## Concept

Every drill in this course asks for the same shape: **write a function called
`solve`**. This page is that shape.

`def` names a new function and lists the **parameters** it expects. The indented
lines below it are its **body** — the work — and `return` is how it hands a
value back to whoever called it:

```python
def double(n):
    """Twice n."""
    return n * 2


print(double(5))
print(double(2.5))
```

Three things to notice. The parameter `n` is a name that does not exist until
the function is called — it takes whatever value the caller passes. The
triple-quoted line is a **docstring**, a sentence saying what the function is
for. And the body is indented; the indentation is what says which lines belong
to the function.

`return` both hands the value back and **ends the call immediately**. Anything
after it never runs:

```python
def first_only(a, b):
    """The first argument, and proof that the second line is dead."""
    return a
    return b


print(first_only("kept", "never reached"))
```

A function with no `return` hands back `None`. This is the quietest bug in early
Python, because nothing goes wrong — you simply get nothing:

```python
def forgot(n):
    """Computes, and then throws it away."""
    n * 3


print(forgot(5))
```

To give more than one answer, return a tuple. One `return`, several values:

```python
def stats(a, b):
    """Sum and product, together."""
    return (a + b, a * b)


print(stats(2, 3))
```

A parameter can carry a **default**, used only when the caller says nothing.
That is what makes one function serve two calls:

```python
def scale(x, times=2):
    """Multiply x, doubling unless told otherwise."""
    return x * times


print(scale(5))
print(scale(5, 10))
```

Functions can also be defined INSIDE other functions, which is how a repeated
step gets a name without leaking out into the rest of the program:

```python
def ends_doubled(values):
    """Double the first and last items, via one local helper."""
    def double(n):
        """Twice n."""
        return n * 2

    return (double(values[0]), double(values[-1]))


print(ends_doubled([1, 2, 7]))
```

## Watch out

- **No `return` means `None`** — the work happens and the answer is discarded.
  A grader sees `None` and reports a failure with nothing obviously wrong.
- **`return` stops the function** — a second `return` on the next line is dead
  code, not a second answer. Return a tuple instead.
- **A default belongs in the `def` line** — `def solve(x, times=2)`. Writing
  `times = 2` in the body ignores whatever the caller passed.

## Worked example

One function, two parameters, a default, and a tuple of two answers.

```python
def summarize(values, places=2):
    """Return (count, mean) with the mean rounded to `places` places."""
    count = len(values)
    mean = round(sum(values) / count, places)
    return (count, mean)


print(summarize([1, 2, 4]))
print(summarize([1, 2, 4], 0))
```

Why each step:

1. The `def` line is the contract: two parameters, the second optional.
2. `count` and `mean` are named inside the body. Those names exist only while
   the call is running.
3. One `return` hands back both answers as a tuple — and the two printed lines
   show the default being used, then overridden.

## Faded practice

### q598
The work is done; `return` is what hands it back.

```python starter
def solve(x):
    """Hand back three times x."""
    _____ x * 3
```

```python solution
def solve(x):
    """Hand back three times x."""
    return x * 3
```

### q599
Two parameters, and both of them have to appear in the body.

```python starter
def solve(a, b):
    """Two parameters, one returned value."""
    return a * _____
```

```python solution
def solve(a, b):
    """Two parameters, one returned value."""
    return a * b
```

## Solo practice

### q600
A default value, used when the caller passes only one argument.

The default lives in the `def` line, and the point of it is that BOTH calls have
to work:

```python worked
def scale(x, times=2):
    """Multiply x, doubling unless told otherwise."""
    return x * times


print("one argument :", scale(7))
print("two arguments:", scale(7, 5))
```

### q601
Two answers from one function — sum and product, in a single return.

## Integrated practice

### q602
A helper function defined inside, then called twice.

### q603
A default that decides which of two orders comes back.

## Misconceptions

- **"The function ran, so it returned something."** — Only `return` returns.
  Without it every call hands back `None`.
- **"Two returns give two answers."** — The first one ends the call. A tuple is
  how one return carries several values.
- **"Parameters are variables I set."** — They are named by the `def` line and
  filled in by the caller, fresh on every call.
