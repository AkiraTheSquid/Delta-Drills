#!/usr/bin/env python3
"""The three standing content guards, in the form a `watch.py` can call.

Each audit already exists as a command-line report. What a Modulario watcher
needs is different: one call, no arguments, raises with a message that says
what to do. Putting that adapter here rather than copying twenty lines into
each folder's `watch.py` matters for a specific reason — a guard copied six
times is a guard that is six different guards inside a month, and the copies
drift toward whichever one was easiest to make pass.

The three rules
---------------
1. **Prerequisites.** Every function, method, attribute and construct used in
   a drill's SOLUTION or in the PROBLEM the learner is handed must be taught
   by a concept at or before that drill's own concept in the lattice. See
   `audit_solution_prereqs.py`. Motivating case: `a.T` drilled on the first
   concept of the course, taught four lessons later.

2. **ARENA grounding.** Every library function we teach or drill must appear
   in the ARENA corpus the course exists to prepare people for. See
   `audit_arena_grounding.py`. Motivating case: 69 drill solutions written in
   `torch.einsum`, which appears in zero of the 458 notebooks.

3. **Symbol coverage.** Every symbol a concept declares in `new_syntax` must
   be drilled at least twice ON THAT CONCEPT. See `audit_symbol_coverage.py`.
   Motivating case: `numpy.random-generator` declares ten symbols and drills
   five of them, across three questions — the mastery models estimate one
   number per concept, so the other five are marked learned on evidence that
   was never collected.

All three are RATCHETS, and that is the whole design
---------------------------------------------------
There is a large existing backlog for each. A hard failure on the current
state would paint every watcher red on day one, and a permanently red watcher
is one nobody reads. So each check fails on violations NOT in its baseline —
which means it goes off exactly when new debt is being added, and stays quiet
about the debt that was already there. Writing a drill that reaches for an
untaught or ungrounded function is what trips it.

Re-recording a baseline to get past a failure is not a fix. It is the
mechanism for admitting debt on purpose, it lands in a diff, and it should
come with a sentence saying why.

Scoping
-------
`kc_prefix` restricts a check to one lesson family, so `lessons/einsum/`'s
watcher reports einsum problems and not the whole bank's. The full-bank check
still runs from `scripts/watch.py` and `lessons/watch.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _prefixes(kc_prefix):
    if not kc_prefix:
        return None
    if isinstance(kc_prefix, str):
        return tuple(p.strip() for p in kc_prefix.split(",") if p.strip())
    return tuple(kc_prefix)


def _scope(violations, kc_prefix):
    pre = _prefixes(kc_prefix)
    if not pre:
        return violations
    return [v for v in violations if (v.get("kc") or "").startswith(pre)]


def check_solution_prereqs(kc_prefix=None) -> None:
    """Every symbol a drill uses must already be taught. Fails on NEW only."""
    import audit_solution_prereqs as A

    known = A.load_baseline()
    if known is None:
        raise AssertionError(
            "solution_prereq_baseline.json is missing — the ratchet cannot tell "
            "new debt from old, so it cannot pass. Re-record it with "
            "audit_solution_prereqs.py --write-baseline")

    violations = _scope(A.find(A.SURFACES), kc_prefix)
    new = sorted({A.key(v) for v in violations} - known)
    where = f" under {','.join(_prefixes(kc_prefix))}" if kc_prefix else ""
    if new:
        raise AssertionError(
            f"{len(new)} drill symbol(s){where} have no prerequisite at or before "
            "their own concept: " + "; ".join(new[:6])
            + (" …" if len(new) > 6 else "")
            + "\n  Every function in a solution or a problem must be taught by a "
              "concept the learner has already reached. Teach it earlier, retag "
              "the drill, or rewrite it to use what the graph has covered.\n"
              "  Inspect: python3 scripts/audit_solution_prereqs.py --new"
            + (f" --kc-prefix {','.join(_prefixes(kc_prefix))}" if kc_prefix else "")
            + "\n  Re-recording the baseline is admitting the debt, not fixing it.")


def check_arena_grounding(kc_prefix=None) -> None:
    """Every library function we teach or drill must exist in ARENA. NEW only."""
    import audit_arena_grounding as G

    known = G.load_baseline()
    if known is None:
        raise AssertionError(
            "arena_grounding_baseline.json is missing — the ratchet cannot tell "
            "new debt from old, so it cannot pass. Re-record it with "
            "audit_arena_grounding.py --write-baseline")

    violations = _scope(G.find(G.SURFACES), kc_prefix)
    new = sorted({G.key(v) for v in violations} - known)
    where = f" under {','.join(_prefixes(kc_prefix))}" if kc_prefix else ""
    if new:
        raise AssertionError(
            f"{len(new)} symbol(s){where} appear in ZERO ARENA notebooks: "
            + "; ".join(new[:6]) + (" …" if len(new) > 6 else "")
            + "\n  ARENA is the source of truth for what is worth teaching. A "
              "function the corpus never uses spends a learner's attention on "
              "something they will never need. Use the operation ARENA actually "
              "writes, or drop it.\n"
              "  Inspect: python3 scripts/audit_arena_grounding.py --new"
            + (f" --kc-prefix {','.join(_prefixes(kc_prefix))}" if kc_prefix else "")
            + "\n  Re-recording the baseline is admitting the debt, not fixing it.")


def check_symbol_coverage(kc_prefix=None) -> None:
    """Every declared symbol is drilled >=2x on its own concept. NEW only."""
    import audit_symbol_coverage as C

    known = C.load_baseline()
    if known is None:
        raise AssertionError(
            "symbol_coverage_baseline.json is missing — the ratchet cannot tell "
            "new debt from old, so it cannot pass. Re-record it with "
            "audit_symbol_coverage.py --write-baseline")

    # Scoping is applied to the VIOLATIONS, not to the measurement: coverage is
    # counted over the whole bank first, because a drill is only evidence for
    # the concept it is tagged to and that tag may sit outside the prefix.
    bad = C.regressions(_scope(C.find(), kc_prefix), known)
    where = f" under {','.join(_prefixes(kc_prefix))}" if kc_prefix else ""
    if bad:
        listed = sorted(bad.values())
        raise AssertionError(
            f"{len(bad)} symbol(s){where} are declared by a concept but drilled "
            f"fewer than {C.MIN_DRILLS_PER_SYMBOL} times on it: "
            + "; ".join(listed[:6]) + (" …" if len(listed) > 6 else "")
            + "\n  A concept's mastery estimate is ONE number covering every "
              "symbol it declares. A symbol with no drills on its own node is "
              "marked learned on evidence about something else. Write the "
              "drills, move the symbol to a concept that drills it, or stop "
              "declaring it.\n"
              "  Inspect: python3 scripts/audit_symbol_coverage.py --new"
            + (f" --kc-prefix {','.join(_prefixes(kc_prefix))}" if kc_prefix else "")
            + "\n  Re-recording the baseline is admitting the debt, not fixing it.")


def check_arena_index_is_current() -> None:
    """The frozen corpus summary must still describe the corpus on disk.

    A guard whose evidence went stale is worse than no guard: it keeps
    answering, confidently, from a corpus that no longer exists. Skipped — not
    failed — when the vendored corpus is absent, because a checkout without
    `arena-book-colab/` is a legitimate state and the frozen index is still
    the right answer in it.
    """
    import audit_arena_grounding as G

    # Deliberately NOT `from audit_arena_frequency import ARENA`: that module
    # exits at import time without torch, which would turn "the corpus is not
    # checked out" into "this watcher cannot run at all" on every bare-python3
    # watcher in the repo. The index knows where its own corpus lives.
    index = G.load_index()
    if not (G.ROOT / index["corpus_root"]).exists():
        return
    G.check_index_fresh()


def run(kc_prefix=None) -> list:
    """The checks a folder `watch.py` should append to its own list."""
    def solution_prereqs_are_taught_first():
        check_solution_prereqs(kc_prefix)

    def every_function_appears_in_arena():
        check_arena_grounding(kc_prefix)

    def every_declared_symbol_is_drilled_twice():
        check_symbol_coverage(kc_prefix)

    return [solution_prereqs_are_taught_first,
            every_function_appears_in_arena,
            every_declared_symbol_is_drilled_twice]
