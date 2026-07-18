---
kc: numpy.structured-dtypes
title: Structured dtypes, text input, datetime64
supporting: [numpy.dtype-astype, numpy.constructors]
new_syntax: [structured-dtype-syntax]
faded: [3]
guided: [96]
independent: [83, 157, 55, 125]
---

## Concept

A dtype doesn't have to be one number. A **structured dtype** gives each
element named **fields** — a typed record, like a C struct or a database row:

> `np.dtype([('name1', type1), ('name2', type2), ...])`
> — a list of (field-name, field-type) pairs, in order.

Field types can themselves be structured (nested records):
`[('position', [('x', float), ('y', float)]), ...]`. Constructors accept
these like any dtype — `np.zeros(n, dtype)` makes n zeroed records — and you
access data **by field name**: `arr['x']` is a view of just that field
across all records (assignable: `arr['x'] = ...` fills one column of the
records). Handy specifics:

- Type spellings: NumPy objects (`np.ubyte`, `float`), or the compact string
  codes (`'u1'` unsigned byte, `'f8'` float64, `'i4'` int32, `'U10'` a
  10-char unicode string).
- Building programmatically from a name list is a comprehension:
  `np.dtype([(n, np.ubyte) for n in names])`.

Two neighbors that travel with structured data:

- **Text input**: `np.genfromtxt(io.StringIO(text), delimiter=',',
  dtype=...)` parses delimited text — including MISSING fields, which it
  fills with NaN (or `filling_values=`). For row-of-strings → typed records,
  build the structured dtype and assign column-by-column via the field
  names.
- **`datetime64`**: dates as an array dtype. `np.datetime64('2016-07')` is a
  month; **arange works on dates** — e.g. all days of a month:
  `np.arange(start_month, next_month, dtype='datetime64[D]')`. The `[D]`,
  `[M]`, `[Y]` suffix is the unit.

Structured arrays are NumPy's answer to "heterogeneous but regular" — when
records outgrow them (mixed missing data, joins, group-bys), that's pandas
territory; the drills stay on the NumPy side.

## Worked example

Task: build an RGBA-byte dtype from field names; create zeroed nested
records and fill one subfield; parse dates.

```python
import numpy as np

# 1. Programmatic structured dtype: one unsigned byte per named field.
names = ['r', 'g', 'b', 'a']
rgba = np.dtype([(n, np.ubyte) for n in names])
assert rgba.names == ('r', 'g', 'b', 'a')
assert rgba.itemsize == 4                      # 4 fields x 1 byte

# 2. Nested records, zero-initialized.
pt = np.zeros(2, dtype=[('position', [('x', float), ('y', float)]),
                        ('color',    [('r', float), ('g', float), ('b', float)])])
# Field access drills down by name; assignment fills across all records.
pt['position']['x'] = [1.0, 2.0]
assert pt['position']['x'].tolist() == [1.0, 2.0]
assert pt['color']['r'].tolist() == [0.0, 0.0]  # untouched fields stay zero

# 3. datetime64: a month is a value; days are an arange between months.
july = np.arange(np.datetime64('2016-07'), np.datetime64('2016-08'),
                 dtype='datetime64[D]')
assert len(july) == 31
assert str(july[0]) == '2016-07-01'
```

Why each step:

1. `itemsize` confirming 4 bytes makes the record concrete: a structured
   element is literally its fields packed in order — dtype-astype's memory
   model, extended.
2. The two-level access `pt['position']['x']` mirrors the two-level dtype;
   note it returns all records' x's at once (a view!) — per-record loops are
   as unnecessary here as anywhere else in NumPy.
3. In the dates step, the month-to-month arange with a `[D]` dtype is the
   idiom for "every day of month m" — the unit suffix does the calendar
   arithmetic (28/29/30/31) for you.

## Faded practice

### q3
Structured dtype of single unsigned bytes, one field per given name.

```python starter
import numpy as np

def solve(names):
    """dtype with each name as an unsigned-byte field, in order."""
    return np.dtype([(n, _____) for n in names])
```

```python solution
import numpy as np

def solve(names):
    """dtype with each name as an unsigned-byte field, in order."""
    return np.dtype([(n, np.ubyte) for n in names])
```

## Guided practice

### q96
1. Zeroed structured array with NESTED fields: position(x, y) and
   color(r, g, b) — the dtype is a list whose field types are themselves
   lists.
2. `np.zeros(n, dtype)` handles initialization once the dtype is right.
3. Match the exact field names and float types; the grader reads
   `arr['position']['x']` etc.

## Independent practice

From the drill bank: q83 (structured grid whose x/y fields hold linspace
coordinates — build zeros, assign fields from meshgrid), q157 (2-D string
array + comma-separated names → typed records: build the dtype, assign
per-field with astype), q55 (every calendar day of a given month — the
datetime64 arange), q125 (CSV text with blank fields → float array with
NaNs: genfromtxt with StringIO).

## Misconceptions

- **"Arrays are numbers-only; records need Python objects."** — Structured
  dtypes store typed records compactly and natively; `dtype=object` is the
  slow last resort, not the tool for regular records.
- **"arr['x'] copies the field out."** — It's a VIEW: assigning to it writes
  into the records. That's what makes field-by-field population of a big
  record array cheap.
- **"Dates need Python's datetime module."** — `datetime64` is vectorized,
  supports arange/comparison/subtraction, and its unit suffix (`[D]`, `[M]`)
  handles calendar arithmetic. Reach for Python datetime only at the
  boundaries.
