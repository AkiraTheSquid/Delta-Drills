#!/usr/bin/env python3
"""Hand-written PyTorch translations for the parked CNN / backprop drills.

These 17 questions belong to no lesson yet — the PyTorch Fundamentals, Autograd
and CNN pages are unwritten, so the KC tags that would reach them do not exist.
They are converted anyway: leaving numpy in the bank means a learner who
reaches them later meets a dialect the whole course has stopped teaching, and
`import numpy as np` in a torch course is exactly the muscle memory the
conversion exists to prevent.

Eleven of them import numpy without ever calling it — a vestigial import, and
the mechanical pass drops it. The rest are listed here only where torch has no
matching spelling.

Same contract as the other manual tables: nothing is trusted on sight.
`verify_equivalence()` runs each translation against the ORIGINAL numpy answer
over identical inputs and refuses the question if they disagree.
"""

MANUAL: dict[int, str] = {
    # sliding_window_view is a numpy stride trick; Tensor.unfold is the torch
    # equivalent, and it wants the window length and step spelled out.
    # A 1-D conv over (batch, channel, length): unfold the LENGTH axis, which
    # is dim 2, not dim 0 — `sliding_window_view(..., axis=2)` said so and the
    # unfold has to say it too.  Window length lands last, matching the k in
    # the subscripts.
    415: """
def solve(x, w):
    return t.einsum('bilk,oik->bol', x.unfold(2, w.shape[2], 1), w)
""",
    # torch.max with no dim returns a 0-d TENSOR (numpy returned a scalar), so
    # the log-sum-exp shift composes the same way but `float()` is what turns
    # it back into a number.
    440: """
def solve(logits, target):
    m = logits.max()
    return float(m + t.log(t.sum(t.exp(logits - m))) - logits[target])
""",
    # np.concatenate splices bare python lists in with the array; torch.cat
    # takes tensors only, so the +/-inf sentinels have to be built as tensors
    # of the input's dtype.
    464: """
def solve(x):
    neg_inf = t.tensor([-t.inf], dtype=x.dtype)
    zero = t.tensor([0.0], dtype=x.dtype)
    pi = t.cat([neg_inf, x, neg_inf])
    pz = t.cat([zero, x, zero])
    return t.maximum(pi[0::2], pi[1::2]), t.maximum(pz[0::2], pz[1::2])
""",
}

EXCLUDE: dict[int, str] = {}

SHADOW_RENAMES: dict[int, tuple[str, str]] = {}

CALL_PATCHES: dict[int, list[tuple[str, str]]] = {}

# None of the parked prompts names a numpy function: they are written around
# the maths (a conv, a softmax, a log-sum-exp) rather than around the library,
# which is why the mechanical pass could take all 17 of them.
TEXT_PATCHES: dict[int, list[tuple[str, str]]] = {}

NO_CROSSCHECK: dict[int, str] = {
    # The log-sum-exp drill deliberately uses logits around 1000 to make the
    # naive exp() overflow.  numpy's array is float64 and torch's tensor is
    # float32, so the two agree only to ~1e-5 relative at that magnitude and
    # the cross-check reads a rounding difference as a disagreement.  The
    # translation is line-for-line identical; what differs is the default
    # dtype, which is the point of the drill's own dtype lesson.  Expected
    # values are still produced by EXECUTING the torch answer, and the real
    # grader verifies them.
    440: "float32 vs float64 log-sum-exp at logits ~1e3 differ by rounding alone",
}
