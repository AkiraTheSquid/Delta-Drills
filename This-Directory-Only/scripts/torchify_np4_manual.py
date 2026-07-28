#!/usr/bin/env python3
"""Hand-written PyTorch translations for the np-4 drills.

Data, not machinery: everything here is a drill whose numpy function has no
torch spelling, so a regex cannot translate it.  Same contract as
`torchify_np23_manual.py` — none of these are trusted on sight;
`verify_equivalence()` runs each one against the ORIGINAL numpy answer over
identical inputs and refuses the question if they disagree.

np-4 is "Applied patterns", and it is the lesson where the two libraries stop
overlapping.  One whole KC (`numpy.structured-dtypes`) has no PyTorch form at
all: a tensor is homogeneous, so record dtypes, `datetime64` and `genfromtxt`
have nothing to translate INTO.  Those drills are excluded and reported rather
than being forced into a shape torch does not have.
"""

MANUAL: dict[int, str] = {
    # --- sliding windows -------------------------------------------------
    # numpy's sliding_window_view is a stride trick; `Tensor.unfold` is the
    # torch equivalent and needs one call per windowed dimension.  The axis
    # order matches: (rows, cols, window_h, window_w).
    167: """
def solve(z, k):
    return z.unfold(0, k, 1).unfold(1, k, 1)
""",
    176: """
def solve(z, bh, bw):
    return z.unfold(0, bh, 1).unfold(1, bw, 1).sum(dim=(2, 3))
""",
    196: """
def solve(x, kern):
    kh, kw = kern.shape
    windows = x.unfold(0, kh, 1).unfold(1, kw, 1)
    return (windows * kern).sum(dim=(2, 3))
""",
    # np.add.reduceat has no torch spelling.  Splitting into fixed-size
    # chunks and summing each is the same operation, and it keeps working
    # when the last block is short.
    195: """
def solve(z, k):
    rows = t.stack([chunk.sum(dim=0) for chunk in z.split(k, dim=0)])
    return t.stack([chunk.sum(dim=1) for chunk in rows.split(k, dim=1)], dim=1)
""",
    # np.pad -> torch.nn.functional.pad, which takes LAST-dim-first pairs.
    209: """
def solve(z, iterations):
    board = z.clone()
    for _ in range(iterations):
        padded = t.nn.functional.pad(board, (1, 1, 1, 1))
        n = (padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
             + padded[1:-1, :-2] + padded[1:-1, 2:]
             + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:])
        board = ((n == 3) | ((board == 1) & (n == 2))).to(t.int64)
    return board
""",
    # --- scatter / gather -------------------------------------------------
    # torch's `repeat` tiles the whole tensor; the numpy meaning — repeat each
    # element its own number of times — is `repeat_interleave`.  The names
    # trade places between the two libraries, so this one cannot be a rule.
    216: """
def solve(c):
    return t.repeat_interleave(t.arange(len(c)), c)
""",
    # --- set combinatorics ------------------------------------------------
    # np.r_ concatenates literals and arrays in one expression; torch.cat takes
    # tensors only, so the literal becomes a one-element tensor.
    143: """
def solve(a):
    mask = t.cat([t.tensor([True]), a[1:] != a[:-1]])
    return a[mask]
""",
    185: """
def solve(a):
    changes = t.nonzero(a[1:] != a[:-1], as_tuple=True)[0] + 1
    starts = t.cat([t.zeros(1, dtype=t.int64), changes])
    values = a[starts]
    counts = t.diff(t.cat([starts, t.tensor([a.numel()])]))
    return values, counts
""",
    # np.apply_along_axis is a loop with a nicer name; torch has no such
    # helper, and `unique` has no per-row form.
    158: """
def solve(z):
    return t.tensor([t.unique(row).numel() for row in z])
""",
    # torch.unique has no return_index, so the first occurrence of each unique
    # row is recovered by scattering positions back and taking the minimum.
    205: """
def solve(z):
    _, inverse = t.unique(z, dim=0, return_inverse=True)
    first = t.full((int(inverse.max()) + 1,), z.shape[0], dtype=t.int64)
    first.scatter_reduce_(0, inverse, t.arange(z.shape[0]), reduce="amin")
    return z[t.sort(first).values]
""",
    # torch.meshgrid REQUIRES an explicit indexing= (numpy defaults to "xy"),
    # so this cannot be a bare rename.
    171: """
def solve(arrays):
    grids = t.meshgrid(*arrays, indexing="ij")
    return t.stack([g.ravel() for g in grids], dim=1)
""",
    105: """
def solve(n, sigma):
    x, y = t.meshgrid(t.linspace(-1, 1, n), t.linspace(-1, 1, n), indexing="xy")
    d2 = x ** 2 + y ** 2
    return t.exp(-d2 / (2.0 * sigma ** 2))
""",
    # --- pairwise metrics -------------------------------------------------
    # torch.corrcoef has no rowvar: it always treats ROWS as the variables, so
    # the column-variable form is a transpose away.
    122: """
def solve(x):
    return t.corrcoef(x.T)
""",
    # --- geometry ---------------------------------------------------------
    # np.c_ glues a column on; torch.cat needs that column to be 2-D already.
    182: """
def solve(pts, m):
    h = t.cat([pts, t.ones(len(pts), 1)], dim=1)
    out = h @ m.T
    return out[:, :2] / out[:, 2:3]
""",
    # --- linear algebra ---------------------------------------------------
    # np.subtract.outer has no torch spelling; the broadcast difference of a
    # column against a row is exactly what it computes.
    118: """
def solve(x, y):
    c = 1.0 / (x[:, None] - y[None, :])
    return t.linalg.det(c)
""",
    # --- memory model -----------------------------------------------------
    # torch has no __array_finalize__.  Metadata survives an operation only if
    # the subclass says so, and __torch_function__ is where that is said —
    # which is also the honest torch answer to "how do I subclass a tensor".
    # `Tensor.name` itself is taken and read-only (it belongs to torch's named
    # -tensor API), so the value is stored in `_name` and re-exposed as a
    # property — the subclass owns the attribute the drill asks for.
    113: """
class NamedTensor(t.Tensor):
    _name = "no name"

    @staticmethod
    def __new__(cls, data, name="no name"):
        obj = t.as_tensor(data).as_subclass(cls)
        obj._name = name
        return obj

    @property
    def name(self):
        return self._name

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        out = super().__torch_function__(func, types, args, kwargs or {})
        if isinstance(out, NamedTensor):
            src = next((a for a in args if isinstance(a, NamedTensor)), None)
            out._name = getattr(src, "_name", "no name")
        return out


def solve(values, name):
    return NamedTensor(values, name)
""",
    # --- random sampling --------------------------------------------------
    # A numpy Generator is not a torch Generator, so every drill taking one as
    # an argument is hand-translated to the torch object and its API:
    # randperm for a without-replacement draw, randint/rand for the rest.
    91: """
def solve(n, p, rng):
    z = t.zeros(n * n)
    z[t.randperm(n * n, generator=rng)[:p]] = 1.0
    return z.reshape(n, n)
""",
    191: """
def solve(x, n_samples, rng):
    idx = t.randint(0, x.numel(), (n_samples, x.numel()), generator=rng)
    means = x[idx].mean(dim=1)
    qs = t.tensor([0.025, 0.975], dtype=means.dtype)
    lo, hi = t.quantile(means, qs)
    return lo, hi
""",
    # torch.argmax refuses a bool tensor where numpy reads it as 0/1.
    200: """
def solve(p, rng):
    u = t.rand((p.shape[0], 1), generator=rng)
    cs = t.cumsum(p, dim=1)
    return (u < cs).to(t.int64).argmax(dim=1)
""",
}


# --------------------------------------------------------------------------
# Excluded by design.  A tensor holds ONE dtype, so numpy's record arrays,
# datetime64 and text parsing have no torch counterpart — there is nothing to
# translate them into, and inventing a torch-flavoured stand-in would teach a
# numpy idea in a dialect that cannot express it.  These stay NumPy and are
# reported at the end of every run.
EXCLUDE = {
    3: "np.dtype record types: tensors are homogeneous, no per-field dtype",
    55: "np.datetime64 has no torch equivalent",
    83: "structured array with named float fields: no torch equivalent",
    96: "nested record dtype: no torch equivalent",
    125: "np.genfromtxt: torch has no text parser",
    157: "np.rec.fromarrays: torch has no record arrays (or string tensors)",
}

# A variable named `t` shadows `import torch as t` and makes the drill
# unrunnable; renamed before translation, since afterwards `t` is the alias.
# np-4 has two: the polar-angle drill calls its angle `t`, and the homography
# drill calls its transform matrix `t`.
SHADOW_RENAMES = {66: ("t", "theta"), 182: ("t", "m")}


# --------------------------------------------------------------------------
# Claims the CALL makes about the result that are spelled numpy-only.  They are
# the point of their drill — a view drill that stops checking the view shares
# storage is not testing anything — so they are re-spelled rather than dropped.
# Not rules: the torch spelling names the specific fixture being compared.
CALL_PATCHES: dict[int, list[tuple[str, str]]] = {
    # torch has no shares_memory; two tensors share storage iff their storage
    # pointers match.  `.data_ptr()` on the TENSOR would compare view offsets
    # instead, which is a different question.
    123: [("bool(np.shares_memory(r, z))",
           "bool(r.untyped_storage().data_ptr() == "
           "z.untyped_storage().data_ptr())")],
    # The subclass check has to name the torch base, or it fails for a reason
    # that has nothing to do with the learner's answer.
    113: [("isinstance(r, np.ndarray)", "isinstance(r, t.Tensor)")],
}


# --------------------------------------------------------------------------
# Prompts that instruct the learner in the dialect the drill no longer grades.
# Nothing executes a prompt, so this is the one kind of leftover numpy the
# cross-check cannot catch — it surfaces in the learner's head, halfway through
# an exercise whose grader is testing something else.
TEXT_PATCHES: dict[int, list[tuple[str, str]]] = {
    # The CALL_PATCH above re-spelled the memory check in torch, and the prompt
    # has to name the same test — otherwise the drill tells the learner it will
    # be graded on a function that is never called.
    123: [("(the grader checks np.shares_memory)",
           "(the grader checks that the result and z report the same "
           "untyped_storage().data_ptr())")],
    # The identity the learner has to make hold is stated in the prompt, so the
    # numpy spelling here is not a passing reference: it is the call the learner
    # would run to check their own work, in a dialect the sandbox no longer
    # teaches.
    216: [("the inverse of np.bincount, so that np.bincount(solve(c), "
           "minlength=len(c)) equals c",
           "the inverse of t.bincount, so that t.bincount(solve(c), "
           "minlength=len(c)) equals c")],
}


# --------------------------------------------------------------------------
# Drills whose inputs cannot be handed to both dialects, so the numpy-vs-torch
# cross-check cannot run.  Each one says why, rather than being silently
# skipped.
NO_CROSSCHECK: dict[int, str] = {
    91: "numpy Generator vs torch.Generator draw different sequences by design",
    191: "numpy Generator vs torch.Generator draw different sequences by design",
    200: "numpy Generator vs torch.Generator draw different sequences by design",
}
