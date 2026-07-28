#!/usr/bin/env python3
"""Hand-written PyTorch translations for the np-2 / np-3 drills.

Data, not machinery: everything here is a drill whose numpy function has no
torch spelling, so a regex cannot translate it.  Kept beside
`torchify_np_drills.py` rather than inside it because the two change for different
reasons — the rules change when a numpy API is discovered, these change one
drill at a time — and because a table of 30-odd rewrites drowns the ~200 lines
of logic that use it.

None of these are trusted on sight: `verify_equivalence()` runs each one
against the ORIGINAL numpy answer over identical inputs and refuses the
question if they disagree.
"""

# Hand-written translations for drills whose numpy function has no torch
# spelling.  Each is checked against the numpy original by verify_equivalence()
# exactly like a regex-translated one — these are *proposals*, not assertions.
#
# The starters are all `return None` stubs, so only the answer body is written
# here; the starter is translated mechanically.

MANUAL: dict[int, str] = {
    # np.indices -> meshgrid.  torch has no single-call index grid.
    11: """
def solve(rows, cols):
    return t.meshgrid(t.arange(rows), t.arange(cols), indexing="ij")[0]
""",
    72: """
def solve(rows, cols, top_left):
    grid = t.stack(t.meshgrid(t.arange(rows), t.arange(cols), indexing="ij"))
    return (grid.sum(dim=0) + top_left) % 2
""",
    # np.pad -> torch.nn.functional.pad, which takes LAST-dim-first pairs.
    17: """
def solve(z):
    return t.nn.functional.pad(z, (1, 1, 1, 1), mode="constant", value=0)
""",
    90: """
def solve(z, fill):
    return t.nn.functional.pad(z, (1, 1, 1, 1), mode="constant", value=fill)
""",
    # torch.nonzero returns an (n, ndim) tensor by default; as_tuple=True is
    # the spelling that matches numpy's tuple-of-index-arrays.
    40: """
def solve(x):
    return t.nonzero(t.as_tensor(x), as_tuple=True)
""",
    # numpy's out= ufuncs are torch's trailing-underscore methods.
    59: """
def solve(a, b):
    b.add_(a)
    a.div_(2)
    a.neg_()
    a.mul_(b)
    return a
""",
    # np.convolve has no torch equivalent; a moving average is a sliding
    # window mean, which unfold expresses directly.
    99: """
def solve(x):
    return x.unfold(0, 3, 1).mean(dim=1)
""",
    # np.divide(..., where=) has no torch equivalent: masked assignment is the
    # torch way to leave the untouched positions alone.
    100: """
def solve(a, b):
    out = t.zeros_like(a)
    nz = b != 0
    out[nz] = a[nz] / b[nz]
    return out
""",
    # Row-mask form: the zero-divisor rows are never handed to the division,
    # so nothing has to be clamped or cleaned up afterwards.
    153: """
def solve(z):
    norms = t.linalg.norm(z, dim=1, keepdim=True)
    out = t.zeros_like(z)
    nz = (norms != 0).squeeze(1)
    out[nz] = z[nz] / norms[nz]
    return out
""",
    162: """
def solve(z):
    sums = z.sum(dim=1, keepdim=True)
    out = t.zeros_like(z)
    nz = (sums != 0).squeeze(1)
    out[nz] = z[nz] / sums[nz]
    return out
""",
    # A numpy Generator has no torch counterpart as an argument; torch.Generator
    # is the equivalent object and randperm is the without-replacement draw.
    101: """
def solve(z, k, gen):
    idx = t.randperm(z.shape[0], generator=gen)[:k]
    return z[idx]
""",
    # np.nditer is a manual broadcast loop; broadcasting is the point.
    111: """
def solve(a, b):
    return a + b
""",
    # np.ogrid has no torch equivalent.  Index vectors shaped (n,1) and (1,n)
    # are the direct translation and stay skinny, which is what ogrid was for;
    # kp-index-grids teaches this spelling.
    116: """
def solve(n):
    y = t.arange(n)[:, None]
    x = t.arange(n)[None, :]
    c = n // 2
    return (y - c).abs() + (x - c).abs()
""",
    # np.argpartition has no torch spelling; topk is the direct replacement
    # and already returns the values largest-first.
    119: """
def solve(z, k):
    return t.topk(z, k).indices
""",
    187: """
def solve(z, k):
    idx = t.topk(z, k, dim=1).indices
    mask = t.zeros_like(z, dtype=t.int64)
    mask.scatter_(1, idx, 1)
    return mask
""",
    194: """
def solve(z, k):
    return t.topk(z, k, dim=1).indices
""",
    206: """
def solve(z, n):
    return t.topk(z, n).values.flip(0)
""",
    # np.take with an index array is plain advanced indexing.
    120: """
def solve(x):
    out = x.clone()
    col_means = t.nanmean(out, dim=0)
    idx = t.nonzero(t.isnan(out), as_tuple=True)
    out[idx] = col_means[idx[1]]
    return out
""",
    # np.result_type takes any number of operands; torch.result_type takes
    # exactly two, and promote_types is the dtype-to-dtype form needed to fold
    # a third one in.
    146: """
def solve(a, b, c):
    dtype = t.promote_types(t.result_type(a, b), c.dtype)
    out = t.empty(a.numel() * 3, dtype=dtype)
    out[0::3] = a
    out[1::3] = b
    out[2::3] = c
    return out
""",
    # np.percentile is torch.quantile on a 0-1 scale.
    147: """
def solve(z, lo_pct, hi_pct):
    qs = t.tensor([lo_pct / 100, hi_pct / 100], dtype=z.dtype)
    lo, hi = t.quantile(z, qs)
    return t.clip(z, lo, hi)
""",
    # torch.unique has no return_index, so first occurrences are found by
    # scattering positions back and taking the minimum per value.
    148: """
def solve(a):
    vals, inverse = t.unique(a, return_inverse=True)
    first = t.full((vals.numel(),), a.numel(), dtype=t.int64)
    first.scatter_reduce_(0, inverse, t.arange(a.numel()), reduce="amin")
    return t.sort(first).values
""",
    # np.maximum.accumulate -> torch.cummax.
    152: """
def solve(z):
    return t.cummax(z, dim=1).values
""",
    # np.apply_along_axis is a loop; torch has no such helper.
    172: """
def solve(z):
    m = int(z.max()) + 1
    return t.stack([t.bincount(row, minlength=m).argmax() for row in z])
""",
    # np.partition -> the median is a sorted-position read; torch.median takes
    # the lower of two middles, which for an odd column count is the median.
    174: """
def solve(z):
    return t.median(z, dim=1).values
""",
    # fill_diagonal_ exists but only on a square tensor, so the off-diagonals
    # are written through arange index pairs instead.
    180: """
def solve(n, d0, d1):
    m = t.zeros(n, n)
    i = t.arange(n)
    m[i, i] = d0
    j = t.arange(n - 1)
    m[j, j + 1] = d1
    m[j + 1, j] = d1
    return m
""",
    # np.intersect1d has no torch spelling; a membership mask over the unique
    # values is the equivalent, and unique already returns them sorted.
    241: """
def solve(a, b):
    ua = t.unique(a)
    return ua[t.isin(ua, b)]
""",
    # sliding_window_view is a numpy stride trick; Tensor.unfold is the torch
    # equivalent and needs the window length and step spelled out.
    117: """
def solve(x, w):
    return x.unfold(0, w, 1)
""",
    175: """
def solve(z, w, step):
    return z.unfold(0, w, 1)[::step]
""",
    # torch.argmax refuses a bool tensor, so the mask is counted as integers.
    130: """
def solve(z):
    mask = z != 0
    return t.where(mask.any(dim=1), mask.int().argmax(dim=1), -1)
""",
    # torch.mean refuses an integer tensor where numpy quietly promotes.
    81: """
def solve(v, reps):
    m = t.tile(v, (reps, 1))
    return m, m.to(t.float32).mean(dim=0)
""",
    # `x.sort()` sorts IN PLACE in numpy and returns None; torch's method
    # returns a sorted copy and leaves x alone, so the drill's whole point —
    # that the caller's array comes back sorted — silently stops holding.
    # torch has no `sort_`, so the sorted values are copied back in.
    235: """
def solve(x):
    x.copy_(x.sort().values)
    return x
""",
    # torch.cumsum requires an explicit dim; numpy flattens when given none.
    234: """
def solve(x):
    return t.cumsum(x, dim=0)
""",
    # torch.quantile needs the levels as a tensor of the input's dtype, where
    # numpy takes any sequence.
    108: """
def solve(x, qs):
    return t.quantile(x, t.tensor(qs, dtype=x.dtype), dim=0)
""",
    166: """
def solve(a, n):
    ret = t.cumsum(a, dim=0, dtype=t.float64)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
""",
}

# torch has no read-only tensor: `ndarray.flags.writeable` guards a numpy-only
# memory model, and the drill's whole content is the ValueError that raises.
# There is nothing to translate it INTO, so it stays in the NumPy dialect and
# is reported rather than quietly dropped.
EXCLUDE = {65: "ndarray.flags.writeable has no torch equivalent (read-only tensors do not exist)"}

# A variable named `t` shadows `import torch as t` and makes the drill
# unrunnable; renamed before translation, since afterwards `t` is the alias.
SHADOW_RENAMES = {236: ("t", "cutoff")}


# --------------------------------------------------------------------------
# Prompts that instruct the learner in the dialect the drill no longer grades.
# Nothing executes a prompt, so this is the one kind of leftover numpy the
# cross-check cannot catch — it surfaces in the learner's head, halfway through
# an exercise whose grader is testing something else.
TEXT_PATCHES: dict[int, list[tuple[str, str]]] = {
    # The drill's promise is that no new tensor is allocated, and in torch it is
    # the trailing-underscore method that keeps that promise; the four ufuncs
    # named here have no torch spelling to reach for at all.
    59: [("use the out= argument of PyTorch's ufuncs (np.add, np.divide, "
          "np.negative, np.multiply)",
          "use PyTorch's in-place methods (add_, div_, neg_, mul_)")],
    # There is no iterator to try: nditer was numpy's way to spell a manual
    # broadcast loop, so the aside pointed at the one thing the torch answer
    # never does.  What the learner has to produce is unchanged.
    111: [("(The classic exercise does this with np.nditer([a, b, None]) — try "
           "the iterator, though any broadcasting approach grades the same.)",
           "(Broadcasting is the whole answer: shapes (m, 1) and (1, n) line up "
           "to (m, n) on their own, so adding the two tensors directly grades "
           "the same as any explicit loop.)")],
}


# --------------------------------------------------------------------------
# Cross-checking a hand-written rewrite needs the numpy inputs converted to
# whatever the torch answer expects.  Two drills take something that is not an
# array, so they say so here rather than being silently skipped.
NO_CROSSCHECK = {
    101: "numpy Generator vs torch.Generator draw different sequences by design",
}
