---
kc: python.dots-and-imports
title: Dots — importing a library, attributes, and methods
supporting: [python.calling-functions, python.lists-and-tuples]
new_syntax: [list.append, math.ceil, math.floor, math.pi, math.sqrt, str.upper, syntax.attribute, syntax.import, syntax.import-as]
faded: [604, 605]
guided: []
independent: [606, 607]
integrated: [608, 609]
---

## Concept

Most of the code you will write reaches into something else with a **dot**. Three
different things use that same dot, and telling them apart is what this page is
for.

**A module.** `import` brings in a library; the dot then reaches inside it for
the things it holds:

```python
import math

print(math.sqrt(9))
print(math.floor(2.7), math.ceil(2.7))
```

`import math as m` gives the same module a shorter name. This is not cosmetic —
it is the convention the rest of this course runs on: the tensor library you
meet in the next lesson is always brought in under a two-letter alias, and every
line of it after that is written through the dot.

```python
import math as m

print(m.sqrt(16))
print(m.floor(-1.2))
```

**An attribute** is a value that belongs to something. It is read with a dot and
**no parentheses**, because there is no work to do — the value is already there:

```python
import math

print(math.pi)
```

**A method** is a function that belongs to something. It is reached with a dot
and, being a function, still has to be **called**:

```python
text = "hello"

print(text.upper())
print(text)
```

Notice `text` is unchanged: `upper()` returned a NEW string. Notice too that the
call is `text.upper()` — the string it works on is the thing before the dot, so
nothing goes in the parentheses.

That is the distinction to hold on to, because it decides whether parentheses
belong:

| Written | Means |
|---|---|
| `math.pi` | an attribute — a value, read as-is |
| `math.sqrt` | the function itself, not run |
| `math.sqrt(9)` | the call — the number 3.0 |
| `text.upper` | the method itself, not run |
| `text.upper()` | the call — the capitalised text |

Not everything is a method. `len` is a plain built-in function that takes the
value as an argument, and strings have no `.len()` at all:

```python
text = "hello"

print(text.upper())
print(len(text))
```

Some methods change the thing they belong to and return `None`. `list.append` is
the one you meet first, and the `None` is what surprises people:

```python
items = [1, 2]
result = items.append(3)

print(items)
print("append returned:", result)
```

So append on its own line, then hand back the LIST — never the result of the
append.

## Watch out

- **Attribute or call?** — `math.pi` has no parentheses because it is a value.
  `math.sqrt(9)` has them because it is work.
- **A method with no parentheses is not run** — returning `text.upper` hands back
  a method object, and nothing complains.
- **`.append()` returns `None`** — it changes the list in place. Returning what
  it gave you returns nothing.

## Worked example

The three kinds of dot in one place, on the same two values.

```python
import math

text = "delta"
x = 6.25

shouted = text.upper()
length = len(text)
root = math.sqrt(x)

print("method on the value  :", shouted)
print("built-in on the value:", length)
print("function in a module :", root)
print("attribute, no call   :", math.pi)
```

Why each step:

1. `text.upper()` is a method: it belongs to the string, and the parentheses run
   it.
2. `len(text)` is a built-in function: the string goes INSIDE the parentheses.
   Same job shape, opposite arrangement.
3. `math.sqrt(x)` reaches into the module, then calls what it found.
4. `math.pi` is read with no parentheses at all, because it is already a value.

## Faded practice

### q604
Reach into the module with a dot, then call what you found.

```python starter
import math


def solve(x):
    """Reach into the module with a dot, then call."""
    return math._____(x)
```

```python solution
import math


def solve(x):
    """Reach into the module with a dot, then call."""
    return math.sqrt(x)
```

### q605
A method belongs to the value — and still has to be called.

```python starter
def solve(text):
    """A method belongs to the value, and still has to be called."""
    return text.upper_____
```

```python solution
def solve(text):
    """A method belongs to the value, and still has to be called."""
    return text.upper()
```

## Solo practice

### q606
One method on the string, one built-in taking the string.

The two look alike and are arranged the opposite way round, which is the whole
point of the drill:

```python worked
text = "abc"

print("the value goes BEFORE the dot:", text.upper())
print("the value goes INSIDE the parentheses:", len(text))
```

### q607
The same module under a shorter name, rounding both ways.

## Integrated practice

### q608
A method, a built-in call, and a module function, in one answer.

### q609
`append` changes the list and hands back `None` — so return the list.

## Misconceptions

- **"Everything after a dot needs parentheses."** — Attributes do not.
  `math.pi()` is an error; `math.pi` is a number.
- **"`text.len()` should work."** — Length is a built-in function, not a string
  method. The arrangement is `len(text)`.
- **"`items.append(x)` gives me the longer list."** — It gives `None` and changes
  `items` in place.
