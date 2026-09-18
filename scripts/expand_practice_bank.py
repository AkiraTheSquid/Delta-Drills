"""Authored September 2026 expansion. Emit append-only patches, never write bank files.

Each family has four distinct tasks and two joint reports. Expectations and
misconception examples are frozen by execution at authoring time. The verifier
uses the exported literals, not recomputed expectations.
"""
import argparse
import ast
import csv
import difflib
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'content-mcp'))
from content_mcp.drills import next_id
from lesson_lib import blank_new_syntax, parse_frontmatter

FAMILIES = []


def task(goal, expression, wrong, why):
    return dict(goal=goal, expression=expression, wrong=wrong, why=why)


def family(kc, args, inputs, fixtures, tasks, prelude='', imports='import torch as t\n', atom=None):
    assert len(tasks) == 4, kc
    FAMILIES.append(dict(kc=kc, args=args, inputs=inputs, fixtures=fixtures,
                         tasks=tasks, prelude=prelude, imports=imports, atom=atom))


def fixtures(args, rows):
    names = [s.strip() for s in args.split(',')]
    return ['\n'.join(f'{name} = {value!r}' for name, value in zip(names, row)) for row in rows]


def grids(shapes=((2, 3), (3, 2), (1, 1), (1, 4)), name='x', extras=None):
    result = []
    for i, shape in enumerate(shapes):
        count = 1
        for size in shape:
            count *= size
        setup = f'{name} = ((t.arange({count}, dtype=t.float64) * {i + 3} + {i}) % 17 - 6).reshape({shape!r})'
        result.append(setup + ('\n' + extras[i] if extras else ''))
    return result


T = task
family('python.values-and-names', 'a, b, c', 'a, b and c are Python integers.',
       fixtures('a,b,c', [(7, 2, 4), (-3, 5, 2), (0, 0, 1), (2, -4, -3)]), [
    T('the change from a to b, as an integer', 'b - a', 'a - b', 'A change is the new value minus the old value.'),
    T('the value obtained by adding b to a, then multiplying that whole result by c', '(a + b) * c', 'a + b * c', 'The multiplier applies to the whole updated value, not just the increment.'),
    T('the integer remaining after removing b groups of c from a', 'a - b * c', '(a - b) * c', 'The removed amount is the group count times its size.'),
    T('a tuple containing the original a and its value after adding b and then c', '(a, a + b + c)', '(a + b, a + b + c)', 'A saved original value must precede both updates.')], imports='')

family('python.types-and-conversion', 'text, scale', 'text is a finite decimal string; scale is a Python integer. Truncation means discarding the fractional part toward zero.',
       fixtures('text,scale', [('3.75', 2), ('-2.8', 3), ('0', 0), ('0.25', -4)]), [
    T('the integer obtained by scaling the numeric value before truncating', 'int(float(text) * scale)', 'int(float(text)) * scale', 'Truncating before multiplication loses fractions that may contribute a whole unit.'),
    T('whether the numeric value is nonzero, as a Boolean', 'bool(float(text))', 'bool(text)', 'A nonempty string containing zero is true as text but false as a number.'),
    T('the numeric value rounded to one decimal place, as a float', 'round(float(text), 1)', 'float(int(float(text)))', 'Rounding to tenths preserves a fractional digit instead of discarding the entire fraction.'),
    T('a tuple of the truncated integer written as text and the original numeric value', '(str(int(float(text))), float(text))', '(text, float(text))', 'Canonical integer text drops the fractional part and redundant decimal notation.')], imports='')

family('python.lists-and-tuples', 'left, right', 'left and right are Python lists of integers; either may be empty. Preserve order and do not mutate them.',
       fixtures('left,right', [([7, 1], [3]), ([], [4, -2]), ([8], []), ([0], [9, 2, 5])]), [
    T('a list containing right followed by left', 'right + left', 'left + right', 'Concatenation order decides which sequence comes first.'),
    T('a two-item tuple holding left and right as separate lists', '(left, right)', 'tuple(left + right)', 'Two containers are not the same as a flat sequence of their entries.'),
    T('the combined number of entries, as an integer', 'len(left) + len(right)', 'len([left, right])', 'The outer pair always has two items; count the entries inside both lists.'),
    T('a list containing two consecutive copies of left followed by right', 'left + left + right', 'left + right + right', 'Only the left sequence is duplicated.')], imports='')

family('python.indexing', 'rows, i, j', 'rows is a nonempty rectangular nested Python list of integers; i and j are valid nonnegative row and column indices.',
       fixtures('rows,i,j', [([[7, 2, -1], [4, 8, 3]], 1, 0), ([[9, 1], [3, -2], [8, 6]], 0, 1), ([[5]], 0, 0), ([[2, 0, 6, 4]], 0, 2)]), [
    T('the entry in row i counted from the right by j positions, with j=0 meaning the last column', 'rows[i][-1-j]', 'rows[i][j]', 'Counting from the right reverses the column position and starts at minus one.'),
    T('column j in the final row', 'rows[-1][j]', 'rows[0][j]', 'The final row is independent of the selected column.'),
    T('a tuple of the first and last entries of row i', '(rows[i][0], rows[i][-1])', '(rows[0][0], rows[-1][-1])', 'Both endpoints belong to the requested row, not opposite corners of the table.'),
    T('a tuple of column j in row i and column j in the first row', '(rows[i][j], rows[0][j])', '(rows[i][j], rows[0][0])', 'The second lookup keeps the column index while changing only the row.')], imports='')

family('python.calling-functions', 'values, adjustment', 'values is a nonempty Python list of integers; adjustment is an integer.',
       fixtures('values,adjustment', [([5, -2, 7], 3), ([4], -2), ([0, 0], 5), ([-4, -9, -1, -2], 1)]), [
    T('the sum after adding adjustment once to the total', 'sum(values) + adjustment', 'sum(values) + adjustment * len(values)', 'The adjustment is applied once to the total, not once per entry.'),
    T('the distance between the largest and smallest values', 'max(values) - min(values)', 'max(values) + min(values)', 'A range measures a difference, even when the minimum is negative.'),
    T('a descending Python list containing all the entries', 'sorted(values, reverse=True)', 'sorted(values)', 'Descending order places the greatest value first.'),
    T('the larger of adjustment and the smallest input value', 'max(adjustment, min(values))', 'min(adjustment, max(values))', 'The inner minimum must be computed before comparing it with the adjustment.')], imports='')

family('python.defining-functions', 'a, b', 'a and b are Python integers.',
       fixtures('a,b', [(8, 3), (-2, 5), (0, 0), (4, -1)]), [
    T('the result of calling helper with a, using its default offset', 'helper(a)', 'helper(a, b)', 'Omitting an argument uses the declared default, not the other outer parameter.'),
    T('the result of calling helper on b with a as its offset', 'helper(b, offset=a)', 'helper(b)', 'An explicit keyword replaces the default value for that call.'),
    T('the result of applying the default-offset helper to a twice in succession', 'helper(helper(a))', 'helper(a) * 2', 'Composition feeds the first return value into the second call; it does not double the result.'),
    T('the result of helper on a minus its result on b, both with their default offset', 'helper(a) - helper(b)', 'helper(a - b)', 'Both independent calls apply their own default offset before subtraction.')],
       prelude='def helper(value, offset=3):\n    return value * 2 + offset', imports='')

family('python.dots-and-imports', 'x, word', 'x is a nonnegative finite Python float; word is a Python string.',
       fixtures('x,word', [(2.3, 'Map'), (0.0, ''), (4.8, 'a b'), (1.0, 'XYZ')]), [
    T('the smallest integer no less than x', 'math.ceil(x)', 'math.floor(x)', 'The upper integer boundary rounds upward, not downward.'),
    T('the area of a circle with radius x, as a float', 'math.pi * x * x', '2 * math.pi * x', 'Area depends on the square of radius; circumference depends on radius once.'),
    T('the uppercase version of word with an exclamation mark appended', 'word.upper() + "!"', 'word + "!"', 'The text operation must change the letters as well as append punctuation.'),
    T('a two-item list containing word and the square root of x', '[word, math.sqrt(x)]', '[word, x * x]', 'A square root undoes a square; multiplying a value by itself does the opposite.')], imports='import math\n')

matrix = grids()
family('torch.tensor-model', 'x', 'x is a nonempty 2-D PyTorch tensor. Return only Python values, not tensors.', matrix, [
    T('the number of entries in two copies of x, as an integer', '2 * x.numel()', '2 * x.ndim', 'Two copies double the entry count, not the number of axes.'),
    T('a tuple containing the column count followed by the row count', '(x.shape[1], x.shape[0])', 'tuple(x.shape)', 'The requested order is columns before rows.'),
    T('a tuple containing x as a nested list and its dtype name as text', '(x.tolist(), str(x.dtype))', '(x.tolist(), str(type(x)))', 'The tensor object type does not describe the type of its stored numbers.'),
    T('a tuple of x as a nested list and its total entry count', '(x.tolist(), x.numel())', '(x.tolist(), x.shape[0])', 'Counting only the outer rows misses every additional column.')])

family('torch.transpose-axes', 'x', 'x is a nonempty 2-D float tensor. Return Python lists or tuples.', matrix, [
    T('the transposed grid with every entry doubled, as a nested list', '(2 * x.T).tolist()', '(2 * x).tolist()', 'Scaling does not exchange rows and columns.'),
    T('a tuple of the transposed grid and its shape as a tuple', '(x.T.tolist(), tuple(x.T.shape))', '(x.T.tolist(), tuple(x.shape))', 'Shape metadata must describe the output layout, not the original layout.'),
    T('the first row of the transposed grid as a flat list', 'x.T[0].tolist()', 'x[0].tolist()', 'A transposed row comes from an original column.'),
    T('the final column of the transposed grid as a flat list', 'x.T[:, -1].tolist()', 'x[:, -1].tolist()', 'A transposed column comes from an original row.')])

family('torch.views-and-copies', 'x', 'x is a nonempty contiguous 2-D float tensor. Return Python values; do not mutate x.', matrix, [
    T('a tuple of transposed values and whether that transposed layout is contiguous', '(x.T.tolist(), x.T.is_contiguous())', '(x.T.tolist(), x.is_contiguous())', 'The layout property belongs to the transposed view, not its source.'),
    T('a tuple of transposed values and whether making that view contiguous preserves its starting address', '(x.T.tolist(), x.T.contiguous().data_ptr() == x.data_ptr())', '(x.T.tolist(), True)', 'A noncontiguous transpose needs a new buffer; singleton layouts may already be contiguous.'),
    T('a tuple of original values and whether asking x itself for contiguous storage preserves its address', '(x.tolist(), x.contiguous().data_ptr() == x.data_ptr())', '(x.tolist(), False)', 'An already contiguous tensor can be returned without allocating another buffer.'),
    T('a tuple of transposed values and whether its contiguous version shares the transpose starting address', '(x.T.tolist(), x.T.contiguous().data_ptr() == x.T.data_ptr())', '(x.T.tolist(), False)', 'A transpose with a size-one axis may already have contiguous reading order.')])

family('torch.constructors', 'rows, cols, value', 'rows and cols are positive integers; value is an integer. All grids must store 64-bit integers. Return nested lists.',
       fixtures('rows,cols,value', [(2, 3, -4), (3, 1, 7), (1, 1, 0), (1, 4, 2)]), [
    T('a rows-by-cols grid filled with value', 't.full((rows, cols), value, dtype=t.int64).tolist()', 't.full((cols, rows), value, dtype=t.int64).tolist()', 'Row count and column count occupy different positions in the shape.'),
    T('a rectangular identity grid of rows rows and cols columns', 't.eye(rows, cols, dtype=t.int64).tolist()', 't.ones((rows, cols), dtype=t.int64).tolist()', 'Only diagonal positions are one; off-diagonal positions remain zero.'),
    T('a rows-by-cols grid with value on the main diagonal and zero elsewhere', '(t.eye(rows, cols, dtype=t.int64) * value).tolist()', 't.full((rows, cols), value, dtype=t.int64).tolist()', 'The supplied value belongs only on the diagonal.'),
    T('a rows-by-cols grid with value off the main diagonal and value+1 on it', '(t.full((rows, cols), value, dtype=t.int64) + t.eye(rows, cols, dtype=t.int64)).tolist()', '(t.eye(rows, cols, dtype=t.int64) * (value + 1)).tolist()', 'Off-diagonal positions keep the background value rather than becoming zero.')])

family('torch.slicing-views', 'x', 'x is a nonempty 2-D integer tensor. Return nested lists; do not modify x.', matrix, [
    T('the grid with both its first row and first column omitted', 'x[1:, 1:].tolist()', 'x[:-1, :-1].tolist()', 'Removing the leading boundaries keeps the lower-right remainder.'),
    T('every second column beginning at column one, retaining every row', 'x[:, 1::2].tolist()', 'x[:, ::2].tolist()', 'The starting offset selects odd column positions rather than even ones.'),
    T('the last row while retaining a size-one row axis', 'x[-1:, :].tolist()', 'x[-1, :].tolist()', 'A slice preserves the row axis; an integer row index removes it.'),
    T('the grid reflected across both its row and column directions', 't.flip(x, (0, 1)).tolist()', 't.flip(x, (0,)).tolist()', 'Reversing rows alone leaves the order within every row unchanged.')])

family('torch.ranges', 'start, count, step', 'start is an integer, count is a positive integer, and step is a nonzero integer. Return flat Python lists.',
       fixtures('start,count,step', [(3, 4, 2), (8, 3, -3), (-2, 1, 5), (0, 5, 1)]), [
    T('count evenly stepped integers, starting one step after start', 't.arange(start + step, start + (count + 1) * step, step).tolist()', 't.arange(start, start + count * step, step).tolist()', 'The first requested point is after the start, not the start itself.'),
    T('count+1 stepped integers including both start and start+count*step', 't.arange(start, start + (count + 1) * step, step).tolist()', 't.arange(start, start + count * step, step).tolist()', 'An inclusive final point requires moving the exclusive boundary one step farther.'),
    T('count equally spaced floating points from start to start+step, inclusive; for count=1 return start', 't.linspace(start, start + step, count).tolist()', 't.linspace(start, start + step * count, count).tolist()', 'Here step determines the whole interval, not the gap between samples.'),
    T('count stepped integers ending at start, in increasing index order with spacing step', 't.arange(start - (count - 1) * step, start + step, step).tolist()', 't.arange(start - count * step, start, step).tolist()', 'The final included value must be start; the incorrect run ends one step early.')])

family('torch.dtype-astype', 'x', 'x is a nonempty 2-D floating tensor with finite small values. Return Python values.',
       [s + '\nx = x / 2' for s in matrix], [
    T('the grid converted to 64-bit integers, as a nested list', 'x.to(t.int64).tolist()', 't.round(x).to(t.int64).tolist()', 'Integer conversion discards the fractional part toward zero rather than rounding to nearest.'),
    T('the storage bytes needed for the same number of entries in 16-bit integers', 'x.numel() * t.tensor(0, dtype=t.int16).element_size()', 'x.numel() * x.element_size()', 'The target element width, not the source width, determines the new storage size.'),
    T('a tuple of integer-converted values and their dtype name', '(x.to(t.int64).tolist(), str(x.to(t.int64).dtype))', '(x.to(t.int64).tolist(), str(x.dtype))', 'The dtype report must describe the converted tensor.'),
    T('the grid truncated to integer values and then stored as 64-bit floats, as a nested list', 'x.to(t.int64).to(t.float64).tolist()', 'x.to(t.float64).tolist()', 'Converting back to floats does not restore discarded fractions.')])

family('torch.reshape-flatten', 'x', 'x is a nonempty contiguous 2-D integer tensor with shape (rows, cols). Return Python lists; preserve row-by-row reading order.', matrix, [
    T('a two-dimensional column containing every entry', 'x.reshape(-1, 1).tolist()', 'x.reshape(1, -1).tolist()', 'A column has one entry per row; a single row has every entry side by side.'),
    T('a flat list of all entries except the final one', 'x.flatten()[:-1].tolist()', 'x[:-1].flatten().tolist()', 'Dropping a final flat entry is not the same as dropping a whole final row.'),
    T('a rows-by-1-by-cols nested list, keeping one middle singleton axis', 'x.reshape(x.shape[0], 1, x.shape[1]).tolist()', 'x.reshape(1, x.shape[0], x.shape[1]).tolist()', 'The singleton belongs between the original axes, not before them.'),
    T('a flat list of positions 1, 3, 5 and so on in reading order', 'x.flatten()[1::2].tolist()', 'x.flatten()[::2].tolist()', 'The requested positions start at one, not zero.')])

family('torch.elementwise-ops', 'x', 'x is a nonempty 2-D float tensor. Return nested Python lists.',
       [s + '\nx = x / 2' for s in matrix], [
    T('each entry restricted to the interval from -1 to 2, inclusive', 'x.clamp(min=-1, max=2).tolist()', 'x.clamp(min=0, max=2).tolist()', 'Negative values down to minus one are allowed and must not all become zero.'),
    T('each entry rounded away from negative infinity to the next integer boundary', 't.ceil(x).tolist()', 't.floor(x).tolist()', 'The upper boundary is selected on both sides of zero.'),
    T('the nonnegative magnitude of every entry, computed as the square root of its square', 't.sqrt(x * x).tolist()', '(x * x).tolist()', 'Squaring removes signs but changes magnitudes; the square root restores their sizes.'),
    T('the fractional remainder after truncating each entry toward zero', '(x - t.trunc(x)).tolist()', '(x - t.floor(x)).tolist()', 'Negative inputs retain negative fractional remainders when truncating toward zero.')])

family('torch.aggregations', 'x', 'x is a nonempty 2-D float tensor. Return Python scalars.', matrix, [
    T('the sum of all entries minus the largest entry', '(x.sum() - x.max()).item()', '(x.sum() - x.min()).item()', 'Remove one occurrence of the largest value, not the smallest.'),
    T('the midpoint between the minimum and maximum', '((x.min() + x.max()) / 2).item()', 'x.mean().item()', 'The midpoint of the extremes ignores how many interior values there are.'),
    T('whether any entry is strictly below the overall mean', '(x < x.mean()).any().item()', '(x < x.mean()).all().item()', 'One qualifying entry is enough; requiring every entry changes the question.'),
    T('the sum of squared deviations from the overall mean', '((x - x.mean()) ** 2).sum().item()', '(x ** 2).sum().item()', 'Deviations are measured relative to the mean, not relative to zero.')])

family('torch.sorting', 'x', 'x is a nonempty 2-D float tensor. Return nested Python lists.', matrix, [
    T('each column sorted descending from top to bottom', 't.sort(x, dim=0, descending=True).values.tolist()', 't.sort(x, dim=1, descending=True).values.tolist()', 'Sorting a column compares different rows while retaining its column position.'),
    T('each row sorted descending from left to right', 't.sort(x, dim=1, descending=True).values.tolist()', 't.sort(x, dim=1).values.tolist()', 'Descending order places the largest row entry first.'),
    T('the original column positions needed to read each row in ascending value order', 't.argsort(x, dim=1).tolist()', 't.sort(x, dim=1).values.tolist()', 'Positions identify where values came from; they are not the sorted values themselves.'),
    T('the largest entry of each row as a size-one inner list', 't.topk(x, 1, dim=1).values.tolist()', 't.topk(x, 1, dim=0).values.tolist()', 'One winner per row requires comparing columns, not comparing rows.')])

random_setups = [s + '\nt.manual_seed(17)' for s in fixtures('rows,cols', [(2, 3), (3, 2), (1, 1), (1, 4)])]
family('torch.random-samplers', 'rows, cols', 'rows and cols are positive integers. Use the already-initialized global random stream without resetting it. Return nested Python lists.', random_setups, [
    T('a rows-by-cols grid of uniform samples in [0,1)', 't.rand(rows, cols).tolist()', 't.randn(rows, cols).tolist()', 'Uniform samples stay between zero and one; normal samples have a different distribution.'),
    T('a rows-by-cols grid of standard-normal samples', 't.randn(rows, cols).tolist()', 't.rand(rows, cols).tolist()', 'Standard-normal values may be negative and are not bounded by one.'),
    T('a rows-by-cols grid of random integers from -3 through 2 inclusive', 't.randint(-3, 3, (rows, cols)).tolist()', 't.randint(-3, 2, (rows, cols)).tolist()', 'The upper sampler boundary is excluded, so it must be beyond the greatest allowed integer.'),
    T('a rows-by-cols grid of uniform samples in [-2,2)', '(4 * t.rand(rows, cols) - 2).tolist()', '(2 * t.rand(rows, cols) - 2).tolist()', 'The interval has width four, not two.')])

seed_setups = fixtures('seed,n', [(3, 4), (18, 2), (0, 1), (91, 5)])
family('torch.random-seeding', 'seed, n', 'seed is a nonnegative integer and n is positive. Each requested sequence starts from a fresh private stream initialized with seed; leave the global stream unchanged. Return flat lists.', seed_setups, [
    T('n uniform values in [1,3)', '(1 + 2 * t.rand(n, generator=t.Generator().manual_seed(seed))).tolist()', 't.rand(n, generator=t.Generator().manual_seed(seed)).tolist()', 'Scaling and shifting must change both boundaries of the interval.'),
    T('n random integers from 2 through 8 inclusive', 't.randint(2, 9, (n,), generator=t.Generator().manual_seed(seed)).tolist()', 't.randint(0, 9, (n,), generator=t.Generator().manual_seed(seed)).tolist()', 'The lower boundary is two; zero and one are not valid outcomes.'),
    T('n normal values with mean -2 and standard deviation 3', '(3 * t.randn(n, generator=t.Generator().manual_seed(seed)) - 2).tolist()', '(t.randn(n, generator=t.Generator().manual_seed(seed)) - 2).tolist()', 'Moving the mean does not also increase the standard deviation.'),
    T('a random ordering of the integers from 1 through n', '(t.randperm(n, generator=t.Generator().manual_seed(seed)) + 1).tolist()', 't.randperm(n, generator=t.Generator().manual_seed(seed)).tolist()', 'The permutation positions begin at zero, but the requested labels begin at one.')])

thread_setups = [s + '\nrng = t.Generator().manual_seed(seed)\nt.rand(3, generator=rng)' for s in seed_setups]
family('torch.random-threading', 'rng, n', 'rng is a caller-owned CPU random generator whose stream may already be advanced; n is positive. Continue that stream without reseeding it or using the global stream. Return flat lists.', thread_setups, [
    T('n uniform values shifted to [-1,0)', '(t.rand(n, generator=rng) - 1).tolist()', '(t.rand(n, generator=t.Generator().manual_seed(0)) - 1).tolist()', 'A newly initialized stream ignores the caller’s seed and previous draws.'),
    T('n random integers from -4 through 4 inclusive', 't.randint(-4, 5, (n,), generator=rng).tolist()', 't.randint(-4, 4, (n,), generator=rng).tolist()', 'An excluded upper boundary of four prevents the valid outcome four.'),
    T('n standard-normal values scaled by two', '(2 * t.randn(n, generator=rng)).tolist()', 't.randn(n, generator=rng).tolist()', 'The stream is correct, but each sampled value still needs the requested scaling.'),
    T('a shuffled ordering of labels 10 through 10+n-1', '(t.randperm(n, generator=rng) + 10).tolist()', '(t.randperm(n, generator=t.Generator().manual_seed(0)) + 10).tolist()', 'Even a permutation must consume the supplied stream rather than resetting another stream.')])

linear_setups = [f'a = t.tensor({a!r}, dtype=t.float64)\nb = t.tensor({b!r}, dtype=t.float64)' for a,b in [
    ([[2, 1], [0, 3]], [4, 6]), ([[3, 0], [1, 2]], [-3, 5]), ([[4]], [8]), ([[1, 2, 0], [0, 2, 1], [0, 0, 3]], [3, -1, 6])]]
family('torch.linalg-basics', 'a, b', 'a is an invertible square float tensor (n,n); b is a float vector (n,). Return Python lists.', linear_setups, [
    T('the solution of a times x equals twice b', 't.linalg.solve(a, 2 * b).tolist()', '(a @ (2 * b)).tolist()', 'Applying the matrix is not the same as undoing its effect.'),
    T('the result of applying a twice to b', '(a @ (a @ b)).tolist()', '((a * a) @ b).tolist()', 'Two matrix applications combine through matrix multiplication, not elementwise squaring.'),
    T('the inverse matrix of twice a, as a nested list', 't.linalg.inv(2 * a).tolist()', '(2 * t.linalg.inv(a)).tolist()', 'Scaling a matrix up scales its inverse down.'),
    T('the solution of the transposed system, a-transpose times x equals b', 't.linalg.solve(a.T, b).tolist()', 't.linalg.solve(a, b).tolist()', 'Transposing coefficients changes the system unless the matrix happens to be symmetric.')])

mask_setups = grids(extras=['low = -2\nhigh = 6','low = 0\nhigh = 5','low = -7\nhigh = -5','low = 2\nhigh = 8'])
family('torch.boolean-masking', 'x, low, high', 'x is a nonempty 2-D numeric tensor; low and high are numbers with low < high. Selection results use row-by-row reading order.', mask_setups, [
    T('a flat list of values strictly between low and high', 'x[(x > low) & (x < high)].tolist()', 'x[(x > low) | (x < high)].tolist()', 'Both bounds must hold simultaneously; accepting either bound admits almost everything.'),
    T('the integer count of entries equal to either boundary', 't.count_nonzero((x == low) | (x == high)).item()', 't.count_nonzero((x == low) & (x == high)).item()', 'One entry cannot equal two distinct boundaries simultaneously.'),
    T('a flat list of entries outside the closed interval from low to high', 'x[(x < low) | (x > high)].tolist()', 'x[(x >= low) & (x <= high)].tolist()', 'The requested selection is the outside, not the inside, of the interval.'),
    T('a nested Boolean list marking values at least low but strictly below high', '((x >= low) & (x < high)).tolist()', '((x > low) & (x <= high)).tolist()', 'The lower endpoint is included and the upper endpoint is excluded.')])

family('torch.argmin-argmax', 'x', 'x is a nonempty 2-D numeric tensor. Positions are zero-based; ties choose the first position. Return Python lists or integers.', matrix, [
    T('the column position of the smallest magnitude in every row', 't.argmin(t.abs(x), dim=1).tolist()', 't.argmin(x, dim=1).tolist()', 'The value closest to zero need not be the most negative one.'),
    T('the row position of the largest magnitude in every column', 't.argmax(t.abs(x), dim=0).tolist()', 't.argmax(t.abs(x), dim=1).tolist()', 'One index per column means comparing rows.'),
    T('the flat reading-order position of the value farthest from zero', 't.argmax(t.abs(x)).item()', 't.argmin(t.abs(x)).item()', 'The farthest magnitude is the largest, not the smallest.'),
    T('the column position of the maximum in each row after reversing that row’s column order', 't.argmax(t.flip(x, (1,)), dim=1).tolist()', 't.argmax(x, dim=1).tolist()', 'The returned positions refer to the reversed row, not the original one.')])

broadcast_setups = [s + '\na = t.arange(x.shape[0], dtype=t.float64) + 2\nb = t.arange(x.shape[1], dtype=t.float64) - 1' for s in matrix]
family('torch.broadcasting-rules', 'x, a, b', 'x is a float tensor (rows,cols); a has shape (rows,) and b has shape (cols,). Return nested Python lists.', broadcast_setups, [
    T('x with a added down rows and b subtracted across columns', '(x + a[:, None] - b[None, :]).tolist()', '(x - a[:, None] + b[None, :]).tolist()', 'The row adjustment is added while the column adjustment is subtracted.'),
    T('the outer table whose entry i,j is a[i] minus b[j]', '(a[:, None] - b[None, :]).tolist()', '(b[:, None] - a[None, :]).tolist()', 'Swapping the operands reverses the sign and exchanges which axis names the rows.'),
    T('x scaled by a once per row, then shifted by b once per column', '(x * a[:, None] + b[None, :]).tolist()', '((x + b[None, :]) * a[:, None]).tolist()', 'The column shift comes after scaling and must not itself be scaled.'),
    T('the outer table of products of a and b, minus x', '(a[:, None] * b[None, :] - x).tolist()', '(x - a[:, None] * b[None, :]).tolist()', 'The residual is the predicted outer table minus the observed grid.')])

family('torch.axis-reductions', 'x', 'x is a nonempty 2-D float tensor. Return Python lists.', matrix, [
    T('each row’s sum retained as a size-one inner list', 'x.sum(dim=1, keepdim=True).tolist()', 'x.sum(dim=0, keepdim=True).tolist()', 'A row total collapses columns and keeps one result for each row.'),
    T('the mean of each column retained inside one outer row', 'x.mean(dim=0, keepdim=True).tolist()', 'x.mean(dim=1, keepdim=True).tolist()', 'Column means compare rows and keep the original column count.'),
    T('x after subtracting its column means', '(x - x.mean(dim=0, keepdim=True)).tolist()', '(x - x.mean(dim=1, keepdim=True)).tolist()', 'Each column needs its own shared baseline; row means center different groups.'),
    T('a flat list of each row’s mean squared value', '(x * x).mean(dim=1).tolist()', '(x.mean(dim=1) ** 2).tolist()', 'Average the squares; squaring the average loses variation within the row.')])

dot_setups = [s + '\nv = t.arange(x.shape[1], dtype=t.float64) + 1' for s in matrix]
family('torch.dot-matmul-patterns', 'x, v', 'x is a float tensor (rows,cols); v is a float vector (cols,). Return Python lists or scalars.', dot_setups, [
    T('the squared length of v as a Python float', 't.dot(v, v).item()', 'v.sum().item()', 'Length squared adds self-products rather than raw coordinates.'),
    T('the dot product of every row of x with v', '(x @ v).tolist()', '(x * v).tolist()', 'A dot product also combines the multiplied coordinates into one value per row.'),
    T('the table of dot products between columns of x', '(x.T @ x).tolist()', '(x @ x.T).tolist()', 'Column comparisons yield a columns-by-columns table; row comparisons yield a different table.'),
    T('v transformed by x and then by the transpose of x', '(x.T @ (x @ v)).tolist()', '(x.T @ x + 1).tolist()', 'The intermediate vector has one value per row before being mapped back to columns.')])

family('torch.stack-concat-interleave', 'x', 'x is a nonempty 2-D numeric tensor (rows,cols). Return nested Python lists.', matrix, [
    T('x followed by its negation along the row direction, shape (2*rows,cols)', 't.cat((x, -x), dim=0).tolist()', 't.cat((x, -x), dim=1).tolist()', 'Row concatenation increases the row count, not the width.'),
    T('each entry paired with its negation in a new final axis, shape (rows,cols,2)', 't.stack((x, -x), dim=2).tolist()', 't.stack((x, -x), dim=0).tolist()', 'The new pair axis belongs inside every entry, not around the whole grid.'),
    T('x, a zero-valued copy, and x again placed side by side', 't.cat((x, t.zeros_like(x), x), dim=1).tolist()', 't.cat((x, t.zeros_like(x), x), dim=0).tolist()', 'Side-by-side panels extend columns while preserving rows.'),
    T('columns alternating an original value and its negation, shape (rows,2*cols)', 't.stack((x, -x), dim=2).reshape(x.shape[0], -1).tolist()', 't.cat((x, -x), dim=1).tolist()', 'Alternating values is different from placing all original columns before all negated columns.')])

ein_imports = 'import torch as t\nimport einops\n'
images = grids(((2, 2, 2, 3), (1, 3, 1, 2), (1, 1, 1, 1), (3, 1, 2, 1)))
family('einops.pattern-language', 'x', 'x is a tensor (batch,channels,height,width). Express the layout using einops; return nested Python lists.', images, [
    T('the layout (width,batch,height,channels)', 'einops.rearrange(x, "b c h w -> w b h c").tolist()', 'einops.rearrange(x, "b c h w -> h b w c").tolist()', 'Width and height have distinct positions even when a square image hides the mistake.'),
    T('the layout (channels,width,batch,height)', 'einops.rearrange(x, "b c h w -> c w b h").tolist()', 'einops.rearrange(x, "b c h w -> c h b w").tolist()', 'The second axis is width, not height.'),
    T('a layout where each channel owns its batch of height-by-width images', 'einops.rearrange(x, "b c h w -> c b h w").tolist()', 'einops.rearrange(x, "b c h w -> b h w c").tolist()', 'Moving channels to the outermost axis differs from moving them to the innermost axis.'),
    T('the layout (height,channels,width,batch)', 'einops.rearrange(x, "b c h w -> h c w b").tolist()', 'einops.rearrange(x, "b c h w -> w c h b").tolist()', 'The first output axis must enumerate source rows, not source columns.')], imports=ein_imports)

family('einops.merge-axes', 'x', 'x is a tensor (batch,channels,height,width). Use einops; return nested Python lists. Within every merged axis the last named factor changes fastest.', images, [
    T('shape (batch*height,channels*width), with batch slow in rows and channel slow in columns', 'einops.rearrange(x, "b c h w -> (b h) (c w)").tolist()', 'einops.rearrange(x, "b c h w -> (h b) (c w)").tolist()', 'Batch-major row blocks keep each image’s rows together.'),
    T('shape (channels,batch*width,height), with batch slow inside the middle axis', 'einops.rearrange(x, "b c h w -> c (b w) h").tolist()', 'einops.rearrange(x, "b c h w -> c (w b) h").tolist()', 'All widths of one batch item precede the next batch item.'),
    T('one flat list in channel, batch, width, height order from slowest to fastest', 'einops.rearrange(x, "b c h w -> (c b w h)").tolist()', 'einops.rearrange(x, "b c h w -> (b c h w)").tolist()', 'A flat list still has a defined factor order; the original reading order is not the requested order.'),
    T('shape (height,width*batch*channels), with channels fastest and width slowest inside each row', 'einops.rearrange(x, "b c h w -> h (w b c)").tolist()', 'einops.rearrange(x, "b c h w -> h (b w c)").tolist()', 'Changing factor order changes which images sit next to one another.')], imports=ein_imports)

split_setups = grids(((2, 12), (1, 6), (1, 1), (3, 4)), extras=['parts = 3','parts = 2','parts = 1','parts = 2'])
family('einops.split-axes', 'x, parts', 'x is a tensor (batch,length); positive integer parts divides length. Use einops; return nested Python lists.', split_setups, [
    T('shape (parts,batch,length/parts), splitting each row into contiguous blocks', 'einops.rearrange(x, "b (p n) -> p b n", p=parts).tolist()', 'einops.rearrange(x, "b (n p) -> p b n", p=parts).tolist()', 'Contiguous blocks place the block index outside the within-block position.'),
    T('shape (batch,length/parts,parts), gathering the same offset from each contiguous block together', 'einops.rearrange(x, "b (p n) -> b n p", p=parts).tolist()', 'einops.rearrange(x, "b (n p) -> b n p", p=parts).tolist()', 'The requested final axis spans blocks, not neighboring original entries.'),
    T('shape (length/parts,batch,parts), interpreting original entries as interleaved parts', 'einops.rearrange(x, "b (n p) -> n b p", p=parts).tolist()', 'einops.rearrange(x, "b (p n) -> n b p", p=parts).tolist()', 'Interleaved parts change fastest in the source row.'),
    T('shape (batch*parts,length/parts), one contiguous source block per output row with batch slowest', 'einops.rearrange(x, "b (p n) -> (b p) n", p=parts).tolist()', 'einops.rearrange(x, "b (p n) -> (p b) n", p=parts).tolist()', 'Batch-major ordering keeps all blocks from one sample together.')], imports=ein_imports)

family('einops.singleton-and-lists', 'x', 'x is a nonempty tensor (rows,cols). Use einops; return nested Python lists.', matrix, [
    T('shape (rows,1,cols,1), inserting two singleton axes', 'einops.rearrange(x, "r c -> r 1 c 1").tolist()', 'einops.rearrange(x, "r c -> 1 r c 1").tolist()', 'The first singleton belongs after the row axis, not before it.'),
    T('shape (cols,rows,1), exchanging existing axes before adding a final singleton', 'einops.rearrange(x, "r c -> c r 1").tolist()', 'einops.rearrange(x, "r c -> r c 1").tolist()', 'Inserting an axis alone does not exchange the original axes.'),
    T('x and its negation as adjacent panels, shape (rows,2*cols), positive panel first', 'einops.rearrange([x, -x], "p r c -> r (p c)").tolist()', 'einops.rearrange([x, -x], "p r c -> r (c p)").tolist()', 'Whole panels must remain together instead of alternating every column.'),
    T('shape (rows,cols,2), pairing each entry with its negation', 'einops.rearrange([x, -x], "p r c -> r c p").tolist()', 'einops.rearrange([x, -x], "p r c -> p r c").tolist()', 'The list axis must become the final pair axis.')], imports=ein_imports)

montages = grids(((6, 1, 2, 3), (4, 2, 1, 2), (1, 1, 1, 1), (2, 1, 3, 1)), extras=['nrows = 2','nrows = 2','nrows = 1','nrows = 1'])
family('einops.grids-montage', 'x, nrows', 'x is an image tensor (batch,channels,height,width); positive nrows divides batch. Arrange images in a row-major grid with nrows image rows unless stated otherwise. Use einops; return nested lists.', montages, [
    T('the montage with channels last, shape (nrows*height,(batch/nrows)*width,channels)', 'einops.rearrange(x, "(r s) c h w -> (r h) (s w) c", r=nrows).tolist()', 'einops.rearrange(x, "(s r) c h w -> (r h) (s w) c", r=nrows).tolist()', 'Row-major image numbering fills one complete image row before moving downward.'),
    T('the montage with channels first and each individual image transposed spatially', 'einops.rearrange(x, "(r s) c h w -> c (r w) (s h)", r=nrows).tolist()', 'einops.rearrange(x, "(r s) c h w -> c (r h) (s w)", r=nrows).tolist()', 'Transposing each tile exchanges its own height and width before assembling the grid.'),
    T('the channels-first montage using column-major image numbering instead', 'einops.rearrange(x, "(s r) c h w -> c (r h) (s w)", r=nrows).tolist()', 'einops.rearrange(x, "(r s) c h w -> c (r h) (s w)", r=nrows).tolist()', 'Column-major numbering advances the image row before moving to the next image column.'),
    T('shape (nrows,channels,height,(batch/nrows)*width), keeping image rows as their own outer axis', 'einops.rearrange(x, "(r s) c h w -> r c h (s w)", r=nrows).tolist()', 'einops.rearrange(x, "(r s) c h w -> r c h (w s)", r=nrows).tolist()', 'Keep each tile’s width contiguous rather than interleaving the tiles pixel by pixel.')], imports=ein_imports)

patches = grids(((2, 2, 4, 6), (1, 1, 2, 6), (1, 1, 1, 1), (1, 2, 4, 2)), extras=['ph = 2\npw = 3','ph = 1\npw = 3','ph = 1\npw = 1','ph = 2\npw = 1'])
family('einops.patches-space-depth', 'x, ph, pw', 'x is a tensor (batch,channels,height,width); positive ph and pw divide height and width. Use nonoverlapping ph-by-pw patches and einops. Return nested lists.', patches, [
    T('shape (batch,patch_rows,patch_cols,channels,ph,pw), keeping every patch axis separate', 'einops.rearrange(x, "b c (h p) (w q) -> b h w c p q", p=ph, q=pw).tolist()', 'einops.rearrange(x, "b c (p h) (q w) -> b h w c p q", p=ph, q=pw).tolist()', 'Patch pixels are contiguous within each block, not interleaved across the image.'),
    T('shape (batch,patch_rows*patch_cols,ph*pw*channels), with channels fastest inside each patch vector', 'einops.rearrange(x, "b c (h p) (w q) -> b (h w) (p q c)", p=ph, q=pw).tolist()', 'einops.rearrange(x, "b c (h p) (w q) -> b (h w) (c p q)", p=ph, q=pw).tolist()', 'Channel-last patch vectors differ from putting an entire channel’s pixels first.'),
    T('shape (batch,channels*ph*pw,patch_rows,patch_cols), with channel slowest inside packed depth', 'einops.rearrange(x, "b c (h p) (w q) -> b (c p q) h w", p=ph, q=pw).tolist()', 'einops.rearrange(x, "b c (h p) (w q) -> b (p q c) h w", p=ph, q=pw).tolist()', 'The packed depth starts with channel blocks, then row and column offsets inside each patch.'),
    T('shape (patch_cols,patch_rows,batch,channels,ph,pw), visiting patch columns before patch rows', 'einops.rearrange(x, "b c (h p) (w q) -> w h b c p q", p=ph, q=pw).tolist()', 'einops.rearrange(x, "b c (h p) (w q) -> h w b c p q", p=ph, q=pw).tolist()', 'The outermost axis must identify patch columns.')], imports=ein_imports)

family('einops.reduce-model', 'x', 'x is a nonempty float tensor (batch,channels,height,width). Use einops; return Python lists.', images, [
    T('the maximum over width for every batch, channel and height position', 'einops.reduce(x, "b c h w -> b c h", "max").tolist()', 'einops.reduce(x, "b c h w -> b c h", "min").tolist()', 'The retained axes are correct, but the requested summary is the maximum.'),
    T('the total over batch and height at each channel and width position', 'einops.reduce(x, "b c h w -> c w", "sum").tolist()', 'einops.reduce(x, "b c h w -> c w", "mean").tolist()', 'A total grows with the number of contributions; an average divides by that count.'),
    T('the mean squared activation per channel', 'einops.reduce(x * x, "b c h w -> c", "mean").tolist()', '(einops.reduce(x, "b c h w -> c", "mean") ** 2).tolist()', 'Squaring before averaging keeps activation energy that cancellation would otherwise remove.'),
    T('the minimum over batch and channels at every spatial position', 'einops.reduce(x, "b c h w -> h w", "min").tolist()', 'einops.reduce(x, "b c h w -> h w", "max").tolist()', 'The same output axes can summarize either extreme; choose the lower one.')], imports=ein_imports)

family('einops.pooling', 'x, ph, pw', 'x is a float tensor (batch,channels,height,width); positive ph and pw divide height and width. Use nonoverlapping ph-by-pw windows and einops. Return nested lists.', patches, [
    T('the minimum value in each window, shape (batch,channels,height/ph,width/pw)', 'einops.reduce(x, "b c (h p) (w q) -> b c h w", "min", p=ph, q=pw).tolist()', 'einops.reduce(x, "b c (h p) (w q) -> b c h w", "max", p=ph, q=pw).tolist()', 'Minimum pooling retains the lowest activation in each local window.'),
    T('the sum of squared values in each window', 'einops.reduce(x * x, "b c (h p) (w q) -> b c h w", "sum", p=ph, q=pw).tolist()', 'einops.reduce(x * x, "b c (h p) (w q) -> b c h w", "mean", p=ph, q=pw).tolist()', 'Local energy is a sum, not an average over the patch area.'),
    T('mean-pooled windows with channels last, shape (batch,height/ph,width/pw,channels)', 'einops.reduce(x, "b c (h p) (w q) -> b h w c", "mean", p=ph, q=pw).tolist()', 'einops.reduce(x, "b c (h p) (w q) -> b c h w", "mean", p=ph, q=pw).tolist()', 'The spatial pooling and the final channel placement are separate layout decisions.'),
    T('the largest squared activation in each window', 'einops.reduce(x * x, "b c (h p) (w q) -> b c h w", "max", p=ph, q=pw).tolist()', '(einops.reduce(x, "b c (h p) (w q) -> b c h w", "max", p=ph, q=pw) ** 2).tolist()', 'A large negative activation can dominate after squaring even if it was not the original maximum.')], imports=ein_imports)

repeat_setups = [s + f'\ncopies = {n}' for s,n in zip(matrix, [2,3,1,2])]
family('einops.repeat-model', 'x, copies', 'x is a tensor (rows,cols); copies is positive. Use einops; return nested lists.', repeat_setups, [
    T('shape (rows,copies,cols), placing a copy axis inside each row', 'einops.repeat(x, "r c -> r p c", p=copies).tolist()', 'einops.repeat(x, "r c -> p r c", p=copies).tolist()', 'The copy axis is inside each row, not outside the whole grid.'),
    T('shape (rows,copies*cols), with complete copies of each row side by side', 'einops.repeat(x, "r c -> r (p c)", p=copies).tolist()', 'einops.repeat(x, "r c -> r (c p)", p=copies).tolist()', 'Tiling whole rows differs from repeating each individual entry.'),
    T('shape (rows*copies,cols), repeating each row consecutively', 'einops.repeat(x, "r c -> (r p) c", p=copies).tolist()', 'einops.repeat(x, "r c -> (p r) c", p=copies).tolist()', 'Consecutive copies of one row must precede the next source row.'),
    T('shape (cols,rows,copies), transposing the grid and adding an innermost copy axis', 'einops.repeat(x, "r c -> c r p", p=copies).tolist()', 'einops.repeat(x, "r c -> r c p", p=copies).tolist()', 'The original column index must become the first output axis.')], imports=ein_imports)

heads = grids(((2, 3, 8), (1, 2, 6), (1, 1, 1), (3, 1, 4)), extras=['heads = 2','heads = 3','heads = 1','heads = 2'])
family('einops.dl-flatten-heads', 'x, heads', 'x is a tensor (batch,time,features); positive heads divides features. Features contain contiguous equal-sized head blocks. Use einops; return nested lists.', heads, [
    T('shape (heads,batch,time,features/heads)', 'einops.rearrange(x, "b t (h d) -> h b t d", h=heads).tolist()', 'einops.rearrange(x, "b t (d h) -> h b t d", h=heads).tolist()', 'Contiguous heads occupy blocks; interleaved heads pick alternating features.'),
    T('shape (batch*heads,time,features/heads), with batch slowest on the first axis', 'einops.rearrange(x, "b t (h d) -> (b h) t d", h=heads).tolist()', 'einops.rearrange(x, "b t (h d) -> (h b) t d", h=heads).tolist()', 'Flattened batch-major heads keep one example’s heads together.'),
    T('shape (batch,time,features), interleaving the head blocks feature-offset first', 'einops.rearrange(x, "b t (h d) -> b t (d h)", h=heads).tolist()', 'x.tolist()', 'The shape stays unchanged, but the feature ordering must change.'),
    T('shape (batch,heads,(features/heads)*time), with time fastest inside each head vector', 'einops.rearrange(x, "b t (h d) -> b h (d t)", h=heads).tolist()', 'einops.rearrange(x, "b t (h d) -> b h (t d)", h=heads).tolist()', 'Each feature’s time sequence must remain contiguous within a head.')], imports=ein_imports)

temporal = grids(((2, 6, 6), (1, 4, 4), (1, 1, 1), (2, 2, 3)), extras=['groups = 2\nwindow = 3','groups = 2\nwindow = 2','groups = 1\nwindow = 1','groups = 1\nwindow = 3'])
family('einops.channel-groups-temporal', 'x, groups, window', 'x is a float tensor (batch,channels,time); groups divides channels and window divides time. Channel groups are contiguous unless stated otherwise. Use einops; return nested lists.', temporal, [
    T('shape (groups,batch,channels/groups,time/window,window), retaining each time window explicitly', 'einops.rearrange(x, "b (g c) (t w) -> g b c t w", g=groups, w=window).tolist()', 'einops.rearrange(x, "b (c g) (t w) -> g b c t w", g=groups, w=window).tolist()', 'Contiguous channel groups are not interleaved channel groups.'),
    T('mean activation of each contiguous channel group in every time window, shape (batch,groups,time/window)', 'einops.reduce(x, "b (g c) (t w) -> b g t", "mean", g=groups, w=window).tolist()', 'einops.reduce(x, "b (g c) (t w) -> b g t", "sum", g=groups, w=window).tolist()', 'Average over both within-group channels and within-window times.'),
    T('shape (batch,channels,time), shuffling contiguous channel groups into interleaved groups', 'einops.rearrange(x, "b (g c) t -> b (c g) t", g=groups).tolist()', 'x.tolist()', 'The channel count stays the same while group membership changes its position.'),
    T('maximum over each time window with channels preserved, shape (batch,channels,time/window)', 'einops.reduce(x, "b c (t w) -> b c t", "max", w=window).tolist()', 'einops.reduce(x, "b c (t w) -> b c t", "mean", w=window).tolist()', 'A temporal maximum keeps the strongest sample rather than averaging the window.')], imports=ein_imports)

out_setups = fixtures('n,start,end', [(4, -2., 4.), (2, 3., -1.), (1, 5., 9.), (3, 0., 2.)])
family('torch.out-argument', 'n, start, end', 'n is positive; start and end are finite numbers. Fill a newly allocated n-by-2 float canvas using output-buffer arguments: its first column samples start to end inclusively, its second samples end to start inclusively. For n=1 each sequence contains only its starting endpoint. Return Python lists.', out_setups, [
    T('the filled n-by-2 canvas', 'canvas.tolist()', 'canvas.T.tolist()', 'The two sequences are columns, not rows.'),
    T('a flat list of first-column minus second-column entries', '(canvas[:, 0] - canvas[:, 1]).tolist()', '(canvas[:, 1] - canvas[:, 0]).tolist()', 'Subtract the reverse sequence from the forward sequence, not vice versa.'),
    T('the filled canvas with both columns doubled', '(2 * canvas).tolist()', '(canvas + 2).tolist()', 'Doubling scales the sampled values; adding two shifts them.'),
    T('a flat row-by-row list alternating the forward and reverse samples', 'canvas.flatten().tolist()', 'canvas.T.flatten().tolist()', 'Alternating sample pairs differs from listing the entire forward sequence first.')],
       prelude='canvas = t.zeros((n, 2))\nt.linspace(start, end, n, out=canvas[:, 0])\nt.linspace(end, start, n, out=canvas[:, 1])')

assign_setups = grids(extras=['value = 8','value = -3','value = 0','value = 5'])
family('torch.slice-assignment', 'x, value', 'x is a nonempty 2-D float tensor; value is a number. On a copy, overwrite the first row with value, then the last column with -value; the column write wins at their intersection. Do not change x. Return Python lists.', assign_setups, [
    T('the edited grid', 'edited.tolist()', 'x.tolist()', 'The returned grid must contain the two writes, not the untouched input.'),
    T('the change at every position, edited minus original', '(edited - x).tolist()', '(x - edited).tolist()', 'A change is measured as the new value minus the old value.'),
    T('the edited first row as a flat list', 'edited[0].tolist()', '(x[0] * 0 + value).tolist()', 'The later column write overrides the first row’s final entry.'),
    T('the edited last column with a size-one column axis retained', 'edited[:, -1:].tolist()', 'edited[:, -1].tolist()', 'Retaining a column axis requires a nested list with one item in each row.')],
       prelude='edited = x.clone()\nedited[0, :] = value\nedited[:, -1] = -value')

ray_setups = [f'r = t.tensor({r!r}, dtype=t.float64)\nu = {u!r}' for r,u in [
    ([[1, -2, 3], [2, 1, -1]], 2.), ([[0, 0, 0], [1, -3, 2]], 0.), ([[-1, 4, 0], [-2, 0, 1]], 0.5), ([[3, 1, -2], [1, 2, 1]], 3.)]]
family('raytracing.ray-parametrisation', 'r, u', 'r is a float tensor (2,3), containing origin then direction; u is nonnegative. The direction is not necessarily unit length. Return flat Python lists.', ray_setups, [
    T('the displacement from the origin to the ray point at parameter u', '(u * r[1]).tolist()', '(r[0] + u * r[1]).tolist()', 'Displacement excludes the origin’s coordinates; position includes them.'),
    T('the midpoint between the origin and the ray point at parameter u', '(r[0] + 0.5 * u * r[1]).tolist()', '(0.5 * (r[0] + u * r[1])).tolist()', 'The origin stays fixed when halving travel along a ray.'),
    T('the point one direction-length beyond parameter u', '(r[0] + (u + 1) * r[1]).tolist()', '(r[0] + u * r[1] + 1).tolist()', 'One further ray step adds the direction vector, not one to every coordinate.'),
    T('the point at parameter u after shifting the origin by one direction vector', '(r[0] + r[1] + u * r[1]).tolist()', '((r[0] + r[1]) * u).tolist()', 'Changing the origin does not multiply that origin by the travel parameter.')])

fan_setups = fixtures('n,limit,distance', [(4, 2., 3.), (2, 1., 0.5), (1, 0., 1.), (3, 3., 2.)])
family('raytracing.make-rays-1d', 'n, limit, distance', 'n is positive, limit is nonnegative, and distance is positive. Rays start at the origin with directions (1,y,0), where y samples -limit to +limit inclusively; one ray uses -limit. Return Python lists.', fan_setups, [
    T('the ray points on the plane x=distance, shape (n,3)', '(distance * rays[:, 1]).tolist()', 'rays[:, 1].tolist()', 'The direction row is the point only on the plane x=1.'),
    T('the y coordinates on the plane x=distance as a flat list', '(distance * rays[:, 1, 1]).tolist()', 'rays[:, 1, 1].tolist()', 'The fan spreads proportionally to distance from the origin.'),
    T('the complete rays with all directions reversed but origins unchanged', 't.stack((rays[:, 0], -rays[:, 1]), dim=1).tolist()', 'rays.tolist()', 'Reversing a ray negates every direction coordinate, including its forward x component.'),
    T('the difference between successive ray directions, shape (n-1,3)', '(rays[1:, 1] - rays[:-1, 1]).tolist()', 'rays[1:, 1].tolist()', 'Neighbor differences remove the shared forward component and measure the angular spacing.')],
       prelude='rays = t.zeros((n, 2, 3))\nrays[:, 1, 0] = 1\nrays[:, 1, 1] = t.linspace(-limit, limit, n)')

family('einops.einsum', 'x', 'x is a nonempty float tensor (batch,channels,height,width). Use einops contractions for the requested sums and products. Return Python lists.', images, [
    T('the squared energy of each batch/channel pair', 'einops.einsum(x, x, "b c h w, b c h w -> b c").tolist()', 'einops.einsum(x, "b c h w -> b c").tolist()', 'Energy accumulates squared activations rather than signed activations.'),
    T('the table of channel dot products for each batch, shape (batch,channels,channels)', 'einops.einsum(x, x, "b c h w, b d h w -> b c d").tolist()', 'einops.einsum(x, x, "b c h w, a c h w -> b a").tolist()', 'The output compares channels within each example, not different examples.'),
    T('the table of width-position dot products per batch after summing over channels and height', 'einops.einsum(x, x, "b c h w, b c h v -> b w v").tolist()', 'einops.einsum(x, x, "b c h w, b c v w -> b h v").tolist()', 'Width positions remain as the two comparison axes; height is reduced.'),
    T('the sum over batch and width at every height/channel pair, shape (height,channels)', 'einops.einsum(x, "b c h w -> h c").tolist()', 'einops.einsum(x, "b c h w -> c h").tolist()', 'The requested output places height before channels.')], imports=ein_imports)

norm_setups = [s + '\nx[0] = 0' for s in matrix]
family('tensor.row-normalization', 'x', 'x is a nonempty float tensor (rows,features). A zero-length row remains zero under normalization. Return Python lists.', norm_setups, [
    T('the change required to make each nonzero row unit length, normalized minus original', '(unit - x).tolist()', '(x - unit).tolist()', 'The correction vector points from the original row toward its normalized row.'),
    T('the squared coordinates of each unit-length row', '(unit * unit).tolist()', 'unit.tolist()', 'Squared coordinates measure each coordinate’s contribution to unit energy and cannot be negative.'),
    T('the sum of the coordinates of every normalized row', 'unit.sum(dim=1).tolist()', 'x.sum(dim=1).tolist()', 'Summation happens after each row receives its own normalization factor.'),
    T('each original row scaled to length two, leaving zero rows zero', '(2 * unit).tolist()', '(2 * x).tolist()', 'Doubling raw rows preserves their differing lengths rather than setting a common length.')],
       prelude='length = x.norm(dim=1, keepdim=True)\nunit = x / t.where(length > 0, length, t.ones_like(length))')

cos_setups = [s + '\nx = x + 0.25\ny = t.flip(x, (0,)) * 2\ny = t.cat((y, -x[:1]), dim=0)' for s in matrix]
family('tensor.cosine-similarity', 'x, y', 'x is a float tensor (queries,features), y is (candidates,features); all rows are nonzero. Return Python lists. Similarity ties select the first candidate.', cos_setups, [
    T('the cosine table mapped from [-1,1] into [0,1]', '((scores + 1) / 2).tolist()', '(scores / 2).tolist()', 'Rescaling the interval needs both a shift and a division.'),
    T('the lowest cosine similarity available for each query', 'scores.min(dim=1).values.tolist()', 'scores.max(dim=1).values.tolist()', 'The least aligned candidate has the lowest score, not the highest.'),
    T('the candidate index most opposed to each query', 'scores.argmin(dim=1).tolist()', 'scores.argmax(dim=1).tolist()', 'Opposition minimizes cosine similarity.'),
    T('the cosine table after reversing every query direction but leaving candidates unchanged', '(-scores).tolist()', 'scores.tolist()', 'Reversing one vector negates its dot product and leaves its length unchanged.')],
       prelude='a = x / x.norm(dim=1, keepdim=True)\nb = y / y.norm(dim=1, keepdim=True)\nscores = a @ b.T')

indexed_setups = [s + '\nids = t.arange(x.shape[0]) % x.shape[1]' for s in matrix]
family('tensor.indexed-selection', 'x, ids', 'x is a nonempty float tensor (rows,cols); ids is an integer vector (rows,) of valid column indices. Return Python lists.', indexed_setups, [
    T('each selected value minus the first entry of its row', '(x.gather(1, ids[:, None])[:, 0] - x[:, 0]).tolist()', '(x.gather(1, ids[:, None])[:, 0] - x[0, 0]).tolist()', 'Every row has its own baseline; a single top-left value cannot replace them all.'),
    T('the selected value from columns counted from the right, with ids=0 meaning the final column', 'x.gather(1, (x.shape[1] - 1 - ids)[:, None])[:, 0].tolist()', 'x.gather(1, ids[:, None])[:, 0].tolist()', 'The supplied positions are right-relative rather than left-relative.'),
    T('each row with its requested entry subtracted from every column', '(x - x.gather(1, ids[:, None])).tolist()', '(x - x[:, :1]).tolist()', 'The row baseline comes from the requested column, not always column zero.'),
    T('the absolute difference between each selected value and that row’s largest value', '(x.gather(1, ids[:, None])[:, 0] - x.max(dim=1).values).abs().tolist()', '(x.gather(1, ids[:, None])[:, 0] - x.min(dim=1).values).abs().tolist()', 'The comparison is against the row maximum, not the row minimum.')])

prob_setups = [s + '\nx = x * 80' for s in matrix]
family('tensor.stable-probabilities', 'x', 'x is a nonempty finite float tensor (batch,classes), possibly with large magnitudes. Compute normalized exponential probabilities stably, one distribution per row. Return Python lists.', prob_setups, [
    T('the probability of the first class in each row', 'probs[:, 0].tolist()', 'probs[:, -1].tolist()', 'The first and last classes are distinct even when both are far from the maximum.'),
    T('one minus each class probability', '(1 - probs).tolist()', 'probs.tolist()', 'The complement measures probability outside that class.'),
    T('the ratio of each class’s unnormalized weight to the largest weight in its row', '(x - x.max(dim=1, keepdim=True).values).exp().tolist()', '((x - x.max(dim=1, keepdim=True).values).exp() + 1).tolist()', 'Dividing by the largest weight is not the same as adding an offset to each ratio.'),
    T('log probabilities normalized independently per row', '(x - x.max(dim=1, keepdim=True).values - weights.sum(dim=1, keepdim=True).log()).tolist()', '(x * 0).tolist()', 'Subtracting only the maximum stabilizes scores but does not complete probability normalization.')],
       prelude='weights = (x - x.max(dim=1, keepdim=True).values).exp()\nprobs = weights / weights.sum(dim=1, keepdim=True)')

class_setups = [s + '\nlabels = (t.arange(x.shape[0]) + 1) % x.shape[1]' for s in matrix]
family('tensor.classifier-evaluation', 'x, labels', 'x is a nonempty float score tensor (batch,classes); labels is a valid integer class vector (batch,). Largest score wins, with first-index tie breaking. Return Python values.', class_setups, [
    T('the list of incorrectly classified example indices in batch order', 't.arange(x.shape[0])[x.argmax(dim=1) != labels].tolist()', 't.arange(x.shape[0])[x.argmax(dim=1) == labels].tolist()', 'The requested indices identify errors rather than correct predictions.'),
    T('the integer number of errors', '(x.argmax(dim=1) != labels).sum().item()', '(x.argmax(dim=1) == labels).sum().item()', 'Count disagreements with the labels, not agreements.'),
    T('each true-label score minus that example’s maximum score', '(x.gather(1, labels[:, None])[:, 0] - x.max(dim=1).values).tolist()', '(x.max(dim=1).values - x.gather(1, labels[:, None])[:, 0]).tolist()', 'This signed margin is nonpositive because the true score cannot exceed the maximum.'),
    T('the predicted labels only for examples classified incorrectly, in batch order', 'x.argmax(dim=1)[x.argmax(dim=1) != labels].tolist()', 'labels[x.argmax(dim=1) != labels].tolist()', 'Report the mistaken predictions rather than the expected labels.')])

cdf_setups = [f'p = t.tensor({p!r}, dtype=t.float64)\nu = t.tensor({u!r}, dtype=t.float64)' for p,u in [
    ([.25, .5, .25], [0., .25, .8]), ([0., .5, .5], [0., .5, .99]), ([1.], [0., .7]), ([.125, .125, .25, .5], [.124, .125, .499, .5])]]
family('tensor.inverse-cdf', 'p, u', 'p is a nonempty vector of nonnegative probabilities summing to one; u is a nonempty vector of samples in [0,1). Class intervals include their left edge and exclude their right edge; zero-width classes are never selected. Return Python lists.', cdf_setups, [
    T('the class index selected by each sample', '(u[:, None] >= p.cumsum(dim=0)[None, :]).sum(dim=1).tolist()', '(u[:, None] > p.cumsum(dim=0)[None, :]).sum(dim=1).tolist()', 'A sample exactly on a boundary belongs to the following nonempty interval.'),
    T('the cumulative mass strictly after each class', '(1 - p.cumsum(dim=0)).tolist()', '(1 - p.cumsum(dim=0) + p).tolist()', 'Mass after a class excludes that class’s own probability.'),
    T('the probability width of each selected class', 'p[(u[:, None] >= p.cumsum(dim=0)[None, :]).sum(dim=1)].tolist()', 'p.cumsum(dim=0)[(u[:, None] >= p.cumsum(dim=0)[None, :]).sum(dim=1)].tolist()', 'An interval’s width is its class probability, not its cumulative right endpoint.'),
    T('each sample’s offset from the left edge of its selected class', '(u - (p.cumsum(dim=0) - p)[(u[:, None] >= p.cumsum(dim=0)[None, :]).sum(dim=1)]).tolist()', '(u - p.cumsum(dim=0)[(u[:, None] >= p.cumsum(dim=0)[None, :]).sum(dim=1)]).tolist()', 'Measure forward from the left edge, not backward from the right edge.')])

family('python.control-flow', 'values, threshold', 'values is a Python list of integers, possibly empty; threshold is an integer. Return Python values.',
       fixtures('values,threshold', [([4, -2, 7, 4], 3), ([], 0), ([0, -1, 2], 0), ([-5, -2, -7], -3)]), [
    T('a list retaining values strictly above threshold, in original order', '[v for v in values if v > threshold]', '[v for v in values if v >= threshold]', 'A strict boundary excludes values equal to the threshold.'),
    T('a list replacing values below threshold with threshold', '[threshold if v < threshold else v for v in values]', '[threshold if v > threshold else v for v in values]', 'Only low values are raised; larger ones must stay unchanged.'),
    T('the sum of positive excesses above threshold', 'sum(v - threshold for v in values if v > threshold)', 'sum(v for v in values if v > threshold)', 'Count only the amount beyond the threshold, not each qualifying value in full.'),
    T('a list of two-item tuples pairing each value with whether it equals threshold', '[(v, v == threshold) for v in values]', '[(v, v > threshold) for v in values]', 'Equality is a separate condition from being above the boundary.')], imports='')

segment_setups = [f'r = t.tensor({r!r}, dtype=t.float64)\ns = t.tensor({s!r}, dtype=t.float64)' for r,s in [
    ([[0,0,0],[2,0,0]], [[4,-1,0],[4,3,0]]),
    ([[0,0,0],[1,0,0]], [[2,0,0],[2,2,0]]),
    ([[0,0,0],[-1,0,0]], [[2,-1,0],[2,1,0]]),
    ([[1,0,0],[1,0,0]], [[0,2,0],[3,2,0]])]]
segment_prelude = '''o, d = r[0, :2], r[1, :2]
a, b = s[0, :2], s[1, :2]
m = t.stack((d, a - b), dim=1)
valid = t.linalg.det(m).abs() >= 1e-8
safe = t.where(valid, m, t.eye(2, dtype=r.dtype))
uv = t.linalg.solve(safe, a - o)
u, v = uv[0], uv[1]
hit = valid & (u >= 0) & (v >= 0) & (v <= 1)'''
family('raytracing.segment-intersection', 'r, s', 'r is a float ray (2,3), origin then direction; s is a float segment (2,3), start then end. Both lie in the xy plane. A determinant magnitude below 1e-8 is singular and counts as a miss. Return Python values.', segment_setups, [
    T('whether the ray hits strictly inside the segment, excluding both endpoints', '(hit & (v > 0) & (v < 1)).item()', 'hit.item()', 'An endpoint hit is allowed for a closed segment but excluded by a strict-interior test.'),
    T('the displacement from ray origin to the hit, as a length-two list; return [0,0] on a miss', 't.where(hit, u * d, t.zeros_like(d)).tolist()', 't.where(hit, v * d, t.zeros_like(d)).tolist()', 'The ray travel parameter and the segment fraction play different roles.'),
    T('the remaining fraction of the segment after the hit, or -1 on a miss', 't.where(hit, 1 - v, t.tensor(-1.)).item()', 't.where(hit, v, t.tensor(-1.)).item()', 'The remaining fraction is measured toward the segment’s end, not from its start.'),
    T('the squared physical travel distance to the hit, or -1 on a miss', 't.where(hit, ((u * d) ** 2).sum(), t.tensor(-1.)).item()', 't.where(hit, u ** 2, t.tensor(-1.)).item()', 'Travel parameters are distances only when the direction has unit length.')], prelude=segment_prelude)

batch_segments = [s.replace('r = t.tensor', 'r = t.tensor').replace('\ns =', '\ns =') + '\nr = t.stack((r, t.stack((r[0], -r[1]))))\ns = t.stack((s, s + t.tensor([1., 0., 0.])))' for s in segment_setups]
batch_segment_prelude = '''o, d = r[:, None, 0, :2], r[:, None, 1, :2]
a, b = s[None, :, 0, :2], s[None, :, 1, :2]
m = t.stack((d + t.zeros_like(a), a - b + t.zeros_like(o)), dim=-1)
valid = t.linalg.det(m).abs() >= 1e-8
safe = t.where(valid[..., None, None], m, t.eye(2, dtype=r.dtype))
uv = t.linalg.solve(safe, (a - o)[..., None])[..., 0]
u, v = uv[..., 0], uv[..., 1]
hit = valid & (u >= 0) & (v >= 0) & (v <= 1)'''
family('raytracing.batched-segments', 'r, s', 'r is a nonempty float ray batch (nr,2,3), origin then direction; s is a nonempty segment batch (ns,2,3), start then end. All geometry is in the xy plane; determinants below 1e-8 in magnitude count as misses. Return Python lists.', batch_segments, [
    T('the strict-interior hit table, shape (nr,ns), excluding segment endpoints', '(hit & (v > 0) & (v < 1)).tolist()', 'hit.tolist()', 'A batch test still needs strict endpoint inequalities when endpoints are excluded.'),
    T('for every segment, the number of rays that hit it', 'hit.sum(dim=0).tolist()', 'hit.sum(dim=1).tolist()', 'One count per segment reduces rays, not segments.'),
    T('the ray travel parameter for each hit and zero for each miss, shape (nr,ns)', 't.where(hit, u, t.zeros_like(u)).tolist()', 't.where(valid, u, t.zeros_like(u)).tolist()', 'An intersection of supporting lines can lie behind a ray or outside a segment.'),
    T('for each ray, the number of segments hit strictly inside their endpoints', '(hit & (v > 0) & (v < 1)).sum(dim=1).tolist()', 'hit.sum(dim=1).tolist()', 'Endpoint contacts do not contribute to an interior-only count.')], prelude=batch_segment_prelude)

fan2_setups = fixtures('ny,nz,yl,zl,depth', [(2,3,1.,2.,3.), (3,1,2.,0.5,2.), (1,1,0.,0.,1.), (1,4,1.,3.,0.5)])
fan2_prelude = '''y = t.linspace(-yl, yl, ny)
z = t.linspace(-zl, zl, nz)
yy = y[:, None] + t.zeros_like(z)[None, :]
zz = t.zeros_like(y)[:, None] + z[None, :]
directions = t.stack((t.ones_like(yy), yy, zz), dim=-1)'''
family('raytracing.make-rays-2d', 'ny, nz, yl, zl, depth', 'ny and nz are positive pixel counts; yl and zl are nonnegative half-extents; depth is positive. Origin-zero rays have direction (1,y,z), sampling both endpoints per axis; a singleton uses the negative endpoint. Flatten pixels with z fastest. Return Python lists.', fan2_setups, [
    T('a (ny,nz,3) grid of ray points on the plane x=depth', '(depth * directions).tolist()', 'directions.tolist()', 'The direction grid reaches x=1; a different depth scales all three coordinates.'),
    T('a flat list of each ray’s y-minus-z coordinate on x=depth', '(depth * (yy - zz)).reshape(-1).tolist()', '(depth * (zz - yy)).reshape(-1).tolist()', 'Coordinate order determines the sign of the requested difference.'),
    T('directions packed into a single (ny*nz,3) list with y fastest instead', 'directions.transpose(0, 1).reshape(-1, 3).tolist()', 'directions.reshape(-1, 3).tolist()', 'Changing the fast pixel axis changes the flattened ray order.'),
    T('a (ny,nz,3) grid of displacement from x=1 to x=depth on each ray', '((depth - 1) * directions).tolist()', '(depth * directions).tolist()', 'The starting plane already accounts for one direction-length of travel.')], prelude=fan2_prelude)

triangle_setups = [f'r = t.tensor({r!r}, dtype=t.float64)\ntr = t.tensor({tr!r}, dtype=t.float64)' for r,tr in [
    ([[0,0,0],[2,0,0]], [[4,-1,-1],[4,3,-1],[4,-1,3]]),
    ([[0,0,0],[1,0,0]], [[2,0,0],[2,2,0],[2,0,2]]),
    ([[0,0,0],[-1,0,0]], [[2,-1,-1],[2,1,-1],[2,-1,1]]),
    ([[0,0,0],[1,0,0]], [[3,0,0],[3,0,0],[3,0,0]])]]
triangle_prelude = '''o, d = r
a, b, c = tr
m = t.stack((-d, b - a, c - a), dim=1)
valid = t.linalg.det(m).abs() >= 1e-8
safe = t.where(valid, m, t.eye(3, dtype=r.dtype))
travel, u, v = t.linalg.solve(safe, o - a)
hit = valid & (travel >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1)'''
family('raytracing.triangle-intersection', 'r, tr', 'r is a float ray (2,3), origin then direction; tr is a float triangle (3,3), vertices A,B,C. Determinant magnitudes below 1e-8 are singular misses. Return Python values.', triangle_setups, [
    T('whether the ray hits strictly inside the triangle, excluding all edges', '(hit & (u > 0) & (v > 0) & (u + v < 1)).item()', 'hit.item()', 'Closed-triangle hits include edges; strict-interior hits exclude every edge.'),
    T('the three barycentric weights in A,B,C order, or three zeros on a miss', 't.where(hit, t.stack((1-u-v, u, v)), t.zeros(3)).tolist()', 't.where(hit, t.stack((u, v, 1-u-v)), t.zeros(3)).tolist()', 'The leftover weight belongs to vertex A, not vertex C.'),
    T('the squared physical travel distance on a hit, or -1 on a miss', 't.where(hit, ((travel*d)**2).sum(), t.tensor(-1.)).item()', 't.where(hit, travel**2, t.tensor(-1.)).item()', 'A non-unit direction turns one parameter unit into more than one distance unit.'),
    T('the displacement from vertex A to the hit, or a zero vector on a miss', 't.where(hit, u*(b-a)+v*(c-a), t.zeros(3)).tolist()', 't.where(hit, o+travel*d, t.zeros(3)).tolist()', 'A displacement from A omits A itself; world coordinates do not.')], prelude=triangle_prelude)

mesh_setups = [s + '\nr = t.stack((r, t.stack((r[0], -r[1]))))\ntr = t.stack((tr, tr + t.tensor([2., 0., 0.])))' for s in triangle_setups]
mesh_prelude = '''o, d = r[:, None, 0, :], r[:, None, 1, :]
a, b, c = tr[None, :, 0, :], tr[None, :, 1, :], tr[None, :, 2, :]
m = t.stack((-d+t.zeros_like(a), b-a+t.zeros_like(o), c-a+t.zeros_like(o)), dim=-1)
valid = t.linalg.det(m).abs() >= 1e-8
safe = t.where(valid[..., None, None], m, t.eye(3, dtype=r.dtype))
suv = t.linalg.solve(safe, (o-a)[..., None])[..., 0]
travel, u, v = suv[..., 0], suv[..., 1], suv[..., 2]
hit = valid & (travel >= 0) & (u >= 0) & (v >= 0) & (u+v <= 1)'''
family('raytracing.mesh-visibility', 'r, tr', 'r is a nonempty float ray batch (nr,2,3), origin then direction; tr is a nonempty triangle batch (nt,3,3), vertices A,B,C. Determinant magnitudes below 1e-8 count as singular misses. Return Python lists.', mesh_setups, [
    T('for each triangle, the number of rays that hit it', 'hit.sum(dim=0).tolist()', 'hit.sum(dim=1).tolist()', 'A triangle-wise count collapses the ray axis.'),
    T('the strict-interior hit table excluding every triangle edge, shape (nr,nt)', '(hit & (u > 0) & (v > 0) & (u+v < 1)).tolist()', 'hit.tolist()', 'Hits on vertices or edges do not count as strict interior hits.'),
    T('the sum of valid hit travel parameters for each ray, counting all hit surfaces and using zero if none hit', 't.where(hit, travel, t.zeros_like(travel)).sum(dim=1).tolist()', 't.where(valid, travel, t.zeros_like(travel)).sum(dim=1).tolist()', 'Solvable supporting-plane intersections can still be behind a ray or outside a triangle.'),
    T('the squared physical travel distance for every valid ray-triangle hit, zero elsewhere', 't.where(hit, travel**2*(d*d).sum(dim=-1), t.zeros_like(travel)).tolist()', 't.where(hit, travel**2, t.zeros_like(travel)).tolist()', 'Distance depends on direction length as well as the travel parameter.')], prelude=mesh_prelude)

stride_setups = grids(((3,4),(4,3),(1,1),(2,5)))
stride_setups[1] += '\nx = x.T'
stride_setups[3] += '\nx = x[:, 1:]'
family('cnn.stride-views', 'x', 'x is a nonempty 2-D float tensor, possibly noncontiguous with a storage offset. Use stride-based views for each layout without changing x. Return Python lists.', stride_setups, [
    T('a transposed grid built from its shape and memory steps', 'x.as_strided((x.shape[1], x.shape[0]), (x.stride(1), x.stride(0))).tolist()', 'x.tolist()', 'Exchanging dimensions also requires exchanging their memory steps.'),
    T('the main diagonal as a flat list', 'x.as_strided((min(x.shape),), (x.stride(0)+x.stride(1),)).tolist()', 'x.as_strided((min(x.shape),), (x.stride(1),)).tolist()', 'Advancing along a diagonal moves one row and one column simultaneously.'),
    T('a grid repeating the first column across every column position', 'x.as_strided(tuple(x.shape), (x.stride(0), 0)).tolist()', 'x.as_strided(tuple(x.shape), (0, x.stride(1))).tolist()', 'A zero column step repeats within each row; a zero row step repeats the first row.'),
    T('every second source row as a grid, starting at row zero', 'x.as_strided(((x.shape[0]+1)//2, x.shape[1]), (2*x.stride(0), x.stride(1))).tolist()', 'x.as_strided(((x.shape[0]+1)//2, x.shape[1]), (x.stride(0), x.stride(1))).tolist()', 'Skipping a source row doubles its memory step.')])

module_setups = [s + f'\nscale = t.tensor({a}, dtype=t.float64)\noffset = t.tensor({b}, dtype=t.float64)' for s,a,b in zip(matrix,[2.,-1.,0.,3.],[1.,4.,-2.,0.])]
module_prelude = '''class Calibrate(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = t.nn.Parameter(scale.clone())
        self.register_buffer("offset", offset.clone())
    def forward(self, values):
        return values * self.weight + self.offset
model = Calibrate()'''
family('cnn.module-state', 'x, scale, offset', 'x is a nonempty 2-D float tensor; scale and offset are scalar float tensors. Build an affine module with trainable weight initialized from scale and fixed buffer offset, whose forward multiplies then shifts. Return nested Python lists.', module_setups, [
    T('the module output for x', 'model(x).tolist()', '(x + scale * offset).tolist()', 'A trainable scale multiplies the input; the fixed offset is added afterward.'),
    T('the change produced by one module application, output minus input', '(model(x) - x).tolist()', '(x - model(x)).tolist()', 'The requested residual is the module’s output minus its input.'),
    T('the result of applying that same module twice', 'model(model(x)).tolist()', '(2 * model(x)).tolist()', 'Composition scales the first offset during the second application; simple doubling does not.'),
    T('the difference between the module outputs for x and for -x', '(model(x) - model(-x)).tolist()', '(model(x) + model(-x)).tolist()', 'Subtracting cancels the fixed offset while adding cancels the input-dependent terms.')], prelude=module_prelude)

linear_layer_setups = [s + '\nw = (t.arange(2*x.shape[1], dtype=t.float64).reshape(2, -1) % 5) - 2\nb = t.tensor([1., -2.], dtype=t.float64)' for s in matrix]
family('cnn.linear-layer', 'x, w, b', 'x is a float tensor (batch,in_features), w is (out_features,in_features), and b is (out_features,). The affine output is the row input multiplied by each weight row, plus b. Return Python lists.', linear_layer_setups, [
    T('the output when the input is negated but weights and bias stay fixed', '(-x @ w.T + b).tolist()', '(-(x @ w.T + b)).tolist()', 'Negating the input leaves the bias sign unchanged.'),
    T('the contribution made by the bias at every output position', '((x @ w.T + b) - (x @ w.T)).tolist()', '(x @ w.T).tolist()', 'The difference between biased and bias-free outputs isolates the bias.'),
    T('the sum of squared affine outputs for each example', '((x @ w.T + b) ** 2).sum(dim=1).tolist()', '((x @ w.T) ** 2).sum(dim=1).tolist()', 'Bias is part of the affine response before output energy is measured.'),
    T('each affine output with its per-example average removed', '((x @ w.T + b) - (x @ w.T + b).mean(dim=1, keepdim=True)).tolist()', '((x @ w.T + b) - (x @ w.T + b).mean(dim=0, keepdim=True)).tolist()', 'Center output features within each example rather than centering each feature across examples.')])

mlp_setups = [s + '\nw1 = (t.arange(3*x[0].numel(), dtype=t.float64).reshape(3, -1) % 5) - 2\nb1 = t.tensor([1., -2., 0.5], dtype=t.float64)\nw2 = t.tensor([[1., -2., 1.], [-1., 1., 2.]], dtype=t.float64)\nb2 = t.tensor([-1., 2.], dtype=t.float64)' for s in images]
family('cnn.mlp', 'x, w1, b1, w2, b2', 'x is a nonempty float image batch (batch,channels,height,width); w1 is (hidden,pixels), b1 is (hidden,), w2 is (outputs,hidden), b2 is (outputs,). Flatten each image, apply the first affine layer and ReLU, then the second affine layer. Return Python lists.', mlp_setups, [
    T('the second layer’s input-dependent contribution with its bias excluded', '(hidden @ w2.T).tolist()', '(hidden @ w2.T + b2).tolist()', 'Excluding the second bias does not mean removing the first layer’s bias.'),
    T('the second-layer outputs after a final ReLU', '(hidden @ w2.T + b2).clamp(min=0).tolist()', '(hidden @ w2.T + b2).tolist()', 'A final ReLU clips negative output scores as well as hidden activations.'),
    T('the number of strictly positive hidden units per example', '(hidden > 0).sum(dim=1).tolist()', '(hidden >= 0).sum(dim=1).tolist()', 'Zero outputs are inactive, not positive.'),
    T('the difference between each example’s greatest and smallest final output scores', '((hidden @ w2.T + b2).max(dim=1).values - (hidden @ w2.T + b2).min(dim=1).values).tolist()', '(hidden @ w2.T + b2).max(dim=1).values.tolist()', 'A score spread compares both extremes, not just the largest score.')],
       prelude='hidden = (x.reshape(x.shape[0], -1) @ w1.T + b1).clamp(min=0)')

conv1_setups = grids(((2,2,5),(1,1,4),(1,1,1),(2,1,6)), extras=['kernel = 2\ns = 1\np = 0','kernel = 3\ns = 2\np = 1','kernel = 1\ns = 1\np = 0','kernel = 2\ns = 2\np = 1'])
conv1_setups = [s + '\nw = (t.arange(2*x.shape[1]*kernel, dtype=t.float64).reshape(2,x.shape[1],kernel)%5)-2' for s in conv1_setups]
conv1_prelude = '''batch, channels, width = x.shape
out_channels, _, kernel = w.shape
padded = t.zeros((batch, channels, width + 2*p), dtype=x.dtype)
padded[:, :, p:p+width] = x
bs, cs, ws = padded.stride()
out_width = 1 + (width + 2*p - kernel)//s
windows = padded.as_strided((batch, channels, out_width, kernel), (bs, cs, s*ws, ws))
response = einops.einsum(windows, w, "b c t k, o c k -> b o t")'''
family('cnn.convolution-1d', 'x, w, s, p', 'x is a float tensor (batch,in_channels,width); w is (out_channels,in_channels,kernel). Use cross-correlation (do not reverse kernels), stride s>=1, and p>=0 zeros at both ends; the kernel fits the padded input. Return Python lists.', conv1_setups, [
    T('the negative part of each convolution response, keeping negative values and replacing positive values with zero', 'response.clamp(max=0).tolist()', 'response.clamp(min=0).tolist()', 'Keeping negative responses is the opposite half of the usual ReLU.'),
    T('the difference between consecutive response positions, shape (batch,out_channels,out_width-1)', '(response[:, :, 1:] - response[:, :, :-1]).tolist()', 'response[:, :, 1:].tolist()', 'A local change subtracts the previous response instead of merely dropping it.'),
    T('the mean squared convolution response per example and output channel', '(response**2).mean(dim=2).tolist()', '(response.mean(dim=2)**2).tolist()', 'Energy averages squared responses before positive and negative values can cancel.'),
    T('each response with its own output-channel mean removed across spatial positions', '(response-response.mean(dim=2,keepdim=True)).tolist()', '(response-response.mean(dim=1,keepdim=True)).tolist()', 'Center each filter’s spatial response, not the set of filters at one position.')], prelude=conv1_prelude, imports=ein_imports)

conv2_setups = grids(((2,2,3,4),(1,1,2,3),(1,1,1,1),(2,1,4,2)), extras=['kh = 2\nkw = 2\ns = 1\np = 0','kh = 1\nkw = 2\ns = 2\np = 1','kh = 1\nkw = 1\ns = 1\np = 0','kh = 2\nkw = 1\ns = 1\np = 1'])
conv2_setups = [s + '\nw = (t.arange(2*x.shape[1]*kh*kw,dtype=t.float64).reshape(2,x.shape[1],kh,kw)%5)-2' for s in conv2_setups]
family('cnn.convolution-2d', 'x, w, s, p', 'x is a float tensor (batch,in_channels,height,width); w is (out_channels,in_channels,kh,kw). Use cross-correlation without reversing kernels, stride s>=1 on both spatial axes, and p>=0 zeros on every side. The kernel fits the padded input. Return Python lists.', conv2_setups, [
    T('the negative part of every convolution response, replacing positive entries with zero', 'response.clamp(max=0).tolist()', 'response.clamp(min=0).tolist()', 'Negative-part clipping retains low responses rather than high responses.'),
    T('the difference between horizontally adjacent output positions', '(response[:, :, :, 1:] - response[:, :, :, :-1]).tolist()', '(response[:, :, 1:, :] - response[:, :, :-1, :]).tolist()', 'Horizontal neighbors differ in width, not in height.'),
    T('the mean squared spatial response per example and output channel', '(response**2).mean(dim=(2,3)).tolist()', '(response.mean(dim=(2,3))**2).tolist()', 'Squaring only after averaging erases spatial variation.'),
    T('the average over output height, retaining output width', 'response.mean(dim=2).tolist()', 'response.mean(dim=3).tolist()', 'Removing height leaves a horizontal profile; removing width leaves a vertical profile.')],
       prelude='response = t.nn.functional.conv2d(x, w, stride=s, padding=p)')

family('cnn.pooling', 'x', 'x is a nonempty float tensor (batch,channels,height,width). Return Python lists.', images, [
    T('global spatial mean after replacing negative pixels with zero', 'x.clamp(min=0).mean(dim=(2,3)).tolist()', 'x.mean(dim=(2,3)).clamp(min=0).tolist()', 'Clipping pixels before averaging prevents negative pixels from cancelling positive ones.'),
    T('global spatial minimum for every example and channel', 'x.min(dim=3).values.min(dim=2).values.tolist()', 'x.max(dim=3).values.max(dim=2).values.tolist()', 'The minimum keeps the weakest spatial activation, not the strongest.'),
    T('the spatial mean of absolute activation per example and channel', 'x.abs().mean(dim=(2,3)).tolist()', 'x.mean(dim=(2,3)).abs().tolist()', 'Measure each magnitude before averaging; signed averages can cancel.'),
    T('the fraction of strictly negative spatial entries per example and channel', '(x<0).to(x.dtype).mean(dim=(2,3)).tolist()', '(x>0).to(x.dtype).mean(dim=(2,3)).tolist()', 'The requested fraction counts negative entries rather than positive ones.')])

batchnorm_setups = [s + '\ng = t.arange(x.shape[1],dtype=t.float64)+1\nb = t.arange(x.shape[1],dtype=t.float64)-2\neps = 0.25' for s in images]
family('cnn.batch-normalization', 'x, g, b, eps', 'x is a nonempty float tensor (batch,channels,height,width); g and b are channel vectors; eps is positive. Use training-mode channel statistics over batch and spatial axes, population variance, and channelwise affine scale g and shift b. Return Python lists.', batchnorm_setups, [
    T('the normalized values before the affine scale and shift', 'normalized.tolist()', 'output.tolist()', 'Normalization and the later affine transform are separate stages.'),
    T('the affine output with its channel shift removed', '(output-b[None,:,None,None]).tolist()', '(output-g[None,:,None,None]).tolist()', 'The shift is the additive channel vector, not the multiplicative scale.'),
    T('the mean squared affine output per channel over batch and space', '(output**2).mean(dim=(0,2,3)).tolist()', '(output.mean(dim=(0,2,3))**2).tolist()', 'Second moments include variance as well as the squared mean.'),
    T('the change from original input to affine normalized output', '(output-x).tolist()', '(x-output).tolist()', 'A change is new output minus original input.')],
       prelude='mean = x.mean(dim=(0,2,3),keepdim=True)\nvariance = ((x-mean)**2).mean(dim=(0,2,3),keepdim=True)\nnormalized = (x-mean)/t.sqrt(variance+eps)\noutput = normalized*g[None,:,None,None]+b[None,:,None,None]')

FIRST_ID = 1363

def _indented(text):
    return '\n'.join('    ' + line if line else '' for line in text.splitlines())

def _source(f, expr, integrated=None):
    body = _indented(f.prelude) if f.prelude else ''
    if integrated is not None or isinstance(expr, tuple):
        pair = integrated if integrated is not None else expr
        body += ('\n' if body else '') + '    first = ' + pair[0] + '\n    second = ' + pair[1] + '\n    return first, second'
    return (body + ('\n' if body else '') + '    return ' + expr)

def _imports(f):
    return '\n'.join(f.imports)

def _run(src, setup, call):
    ns = {}
    exec(_imports(setup) if hasattr(setup, 'imports') else '', ns)
    exec(setup, ns)
    exec(src, ns)
    return eval(call, ns)

def _lesson_paths():
    out = {}
    for p in (ROOT / 'Local_Deployed_Shared' / 'lessons').rglob('kp-*.md'):
        m = re.search(r'^kc:\s*([^\n]+)', p.read_text(), re.M)
        if m: out[m.group(1).strip()] = p
    return out

def _fm_list(text, key, ids):
    pat = re.compile(r'^' + re.escape(key) + r':\s*\[([^\n]*)\]$', re.M)
    val = ', '.join(str(x) for x in ids)
    if pat.search(text): return pat.sub(key + ': [' + val + ']', text, count=1)
    return text.replace('---\n\n', key + ': [' + val + ']\n---\n\n', 1)

def _lesson_patch(path, solo, integrated):
    old = path.read_text(); text = old
    fm, _ = parse_frontmatter(old, path)
    ind = list(fm.get('independent', [])) + solo
    integ = list(fm.get('integrated', [])) + integrated
    text = _fm_list(text, 'independent', ind)
    text = _fm_list(text, 'integrated', integ)
    def block(title, ids):
        return '\n'.join([f'### {title}'] + [f'- Practice problem `{i}`: solve the function for the supplied cases.' for i in ids]) + '\n\n'
    solo_block = block('Additional solo practice', solo)
    integ_block = block('Additional integrated practice', integrated)
    marker = re.search(r'^##\s+(?:Integrated practice|Misconceptions)\b', text, re.M)
    pos = marker.start() if marker else len(text)
    text = text[:pos].rstrip() + '\n\n' + solo_block + text[pos:]
    marker = re.search(r'^##\s+Misconceptions\b', text, re.M)
    pos = marker.start() if marker else len(text)
    text = text[:pos].rstrip() + '\n\n## Integrated practice\n\n' + integ_block + text[pos:]
    return old, text

def build_patch():
    questions = json.loads((ROOT / 'This-Directory-Only' / 'questions_full.json').read_text())
    cross = json.loads((ROOT / 'Local_Deployed_Shared' / 'concept-graph' / 'kc_atom_crosswalk.json').read_text())
    qmatrix = json.loads((ROOT / 'Local_Deployed_Shared' / 'lessons' / 'qmatrix_tags.json').read_text())
    by_kc = {}
    for q in questions:
        for tag in qmatrix.get(str(q['id']), {}).get('knowledge_components', []): by_kc.setdefault(tag, q)
    paths = _lesson_paths(); records=[]; tag_lines=[]; csv_rows=[]; lessons={}; qid=FIRST_ID
    for f in FAMILIES:
        kc,args,prelude,imports,fixtures_list,tasks,prompt = (f['kc'],f['args'],f['prelude'],f['imports'],f['fixtures'],f['tasks'],f['inputs'])
        if kc not in paths: raise RuntimeError('missing lesson for '+kc)
        solo_ids=[]; int_ids=[]
        specs=[(x, None) for x in tasks]
        specs += [(None, (tasks[0], tasks[1])), (None, (tasks[2], tasks[3]))]
        for idx,(single,pair) in enumerate(specs):
            if pair:
                a,b=pair; expr=(a['expression'],b['expression']); wrong=(a['wrong'],b['expression'])
                goal='return a tuple: first ' + a['goal'] + '; second ' + b['goal']
                why=a['why']+' Also '+b['why'][0].lower()+b['why'][1:]
            else:
                expr=single['expression']; wrong=single['wrong']; goal=single['goal']; why=single['why']
            class F: pass
            ff=F(); ff.prelude=prelude; ff.imports=imports
            if expr is None: raise RuntimeError(f'empty expression {kc} stage {idx} pair={pair!r}')
            if isinstance(expr, tuple) and len(expr)==2: expr = '(' + expr[0] + ', ' + expr[1] + ')'
            src = imports + ('\n' if imports else '') + 'def solve(' + args + '):\n' + _source(ff, expr) + '\n'
            if wrong is None: raise RuntimeError(f'empty wrong {kc} stage {idx}')
            if isinstance(wrong, tuple) and len(wrong)==2: wrong = '(' + wrong[0] + ', ' + wrong[1] + ')'
            wrong_src = imports + ('\n' if imports else '') + 'def solve(' + args + '):\n' + _source(ff, wrong) + '\n'
            cases=[]; wrong_examples=[]
            for setup in fixtures_list:
                call='solve(' + args + ')'
                ns={}; exec(imports,ns); exec(setup,ns); exec(src,ns); expected=eval(call,ns)
                cases.append({'setup_code': imports+'\n'+setup, 'call': call, 'expected_expr': repr(expected), 'assert_code': 'assert type(result) is '+type(expected).__name__})
                try:
                    ns={}; exec(imports,ns); exec(setup,ns); exec(wrong_src,ns); got=eval(call,ns)
                    if got != expected and not wrong_examples: wrong_examples=[{'setup_code': imports+'\n'+setup, 'call': call, 'actual_expr': repr(got), 'explanation': why}]
                except Exception: pass
            if not wrong_examples: raise RuntimeError(f'no wrong example {kc} stage {idx}')
            q={'id':qid,'topic':by_kc.get(kc,{}).get('topic','Python'),'subtopic':by_kc.get(kc,{}).get('subtopic','Core practice'),'subtopic_key':by_kc.get(kc,{}).get('subtopic_key',''), 'question_text':'Given '+prompt+' Return '+goal+'.', 'answer_code':src, 'difficulty_score':62 if pair else 48, 'difficulty_label':'medium', 'expected_output':'', 'language':'python','primary_library':'torch','task_type':'code_completion','function_name':'solve','starter_code':imports+'\n'+'def solve('+args+'):\n    pass\n','test_cases':cases,'submission_mode':'function','wrong_examples':wrong_examples,'expected_artifact_type':'python_value','supports_visual_output':False,'source_type':'authored_expansion','source_path':'scripts/expand_practice_bank.py'}
            records.append(q); solo_ids.append(qid) if not pair else int_ids.append(qid)
            atom=(cross.get('kcs',{}).get(kc,{}).get('atoms') or [{}])[0].get('a','')
            if atom: tag_lines.append(json.dumps({'question_id':qid,'atom_id':atom,'confidence':0.95,'source':'authored_expansion'}))
            csv_rows.append({'id':qid,'topic':q['topic'],'subtopic':q['subtopic'],'question_text':q['question_text'],'answer_code':q['answer_code'],'difficulty_score':q['difficulty_score'],'difficulty_label':q['difficulty_label'],'source_type':'authored_expansion','source_path':q['source_path']})
            qid += 1
        lessons[kc]=(solo_ids,int_ids)
    if qid != FIRST_ID + len(records): raise AssertionError('id drift')
    files={}
    for rel, data in [('This-Directory-Only/questions_full.json', questions+records), ('Local_Deployed_Shared/questions.json', questions+records)]: files[rel]=json.dumps(data,indent=2)+'\n'
    for rel in ['This-Directory-Only/csv files of problems/curated_additions.csv']:
        p=ROOT/rel; old_rows=list(csv.DictReader(p.open())); headers=['Topic','Subtopic','Question','Answer','Problem difficulty','Output']; buf=io.StringIO(); w=csv.DictWriter(buf,fieldnames=headers); w.writeheader(); w.writerows(old_rows)
        for row in csv_rows: w.writerow({'Topic':row['topic'],'Subtopic':row['subtopic'],'Question':row['question_text'],'Answer':row['answer_code'],'Problem difficulty':row['difficulty_label'],'Output':''})
        files[rel]=buf.getvalue()
    tagp=ROOT/'This-Directory-Only/backend/app/data/question_atom_tags.jsonl'; files[str(tagp.relative_to(ROOT))]=tagp.read_text().rstrip()+'\n'+'\n'.join(tag_lines)+'\n'
    for kc,(s,i) in lessons.items():
        old,new=_lesson_patch(paths[kc],s,i); files[str(paths[kc].relative_to(ROOT))]=new
    patch=['*** Begin Patch']
    for path,new in files.items():
        old=(ROOT/path).read_text()
        if old==new: continue
        patch += ['*** Update File: '+path]
        diff=list(difflib.unified_diff(old.splitlines(),new.splitlines(),n=2)); patch += [line for line in diff[2:] if not line.startswith(('---','+++'))]
    patch += ['*** End Patch']; return '\n'.join(patch)

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--emit-patch',action='store_true'); args=ap.parse_args()
    if not args.emit_patch: raise SystemExit('use --emit-patch; apply output with patch tool')
    print(build_patch())
