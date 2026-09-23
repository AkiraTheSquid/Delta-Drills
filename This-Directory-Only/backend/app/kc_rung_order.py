"""Rung-ordering helpers over a KC's question ids — split out of kc_graph.py
(2026-09-22, Modulario size gate). kc_graph re-exports both names, so
`kc_graph.with_example_first` / `kc_graph.lowest_rung` keep working;
`lowest_rung_by` takes the rank function so this module never imports kc_graph.
"""

from __future__ import annotations

from typing import Callable, Iterable, List

from app import lessons


def with_example_first(qids: Iterable[int]) -> List[int]:
    """The rung's example-bearing drills, if it still has any unserved.

    THE FADE FROM `partial` TO `solo`, and it needs no schedule and no counter.
    A solo drill carries an example only when its KP authored one for it (six of
    nineteen on `torch.tensor-model`), and the queue never repeats a served
    question — so serving the example-bearing ones first means a learner meets
    an example, then a few unaided problems, then another example introducing
    the next move, and then none at all once they are spent. "It shows examples
    less and less until it doesn't show any examples at all."

    Returns the whole input when no example-bearing drill is left, which is the
    far end of that fade rather than a failure.
    """
    pool = list(qids)
    with_example = [q for q in pool if lessons.has_worked_example(q)]
    return with_example or pool


def lowest_rung_by(qids: Iterable[int], rank: Callable[[int], int]) -> List[int]:
    """The questions on the least-scaffolded rung still available.

    Called with the questions a learner can actually be served right now, so an
    exhausted faded rung falls through to guided and then independent instead of
    dead-ending. Returns [] for an empty input, which callers read as "this KC
    has nothing left" rather than as an ordering result.
    """
    pool = list(qids)
    if not pool:
        return []
    floor = min(rank(q) for q in pool)
    return [q for q in pool if rank(q) == floor]
