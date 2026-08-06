"""Is the data the lattice runs on actually inside this image?

`kc_graph` degrades quietly when one of its data files is missing: `_read_json`
logs a warning, returns `{}`, and every caller carries on. The result is not an
error anywhere — it is a lattice in which all 63 concepts report the bare BKT
prior with tier `unmapped` and covered weight 0, which is indistinguishable
from a learner who has never practised. The knowledge graph then draws one flat
colour and nothing the learner does moves it.

That is not hypothetical. `Local_Deployed_Shared/concept-graph/` is gitignored
and the Dockerfile never copied `kc_atom_crosswalk.json` into the image, so the
production backend ran with zero mapped concepts while the account behind it
held 95 atom posteriors and 41 graded attempts in one subtopic. The only
symptom was a WARNING line in the Fly log.

So the counts go on `/health`, where an empty join is one unauthenticated
request away from being visible instead of a month away. Deliberately counts
rows rather than checking `path.exists()`: a file that is present but truncated
to `{}` fails exactly the same way and must read the same.
"""
from __future__ import annotations

from app import kc_graph


def lattice_data_health() -> dict[str, int]:
    """Row counts for the three files `kc_graph` needs, as loaded (cached).

    Reaches for `kc_graph`'s private readers on purpose — they are the
    lru_cached loaders every other caller goes through, so these numbers are
    what the serving path actually sees, not a second read that could disagree
    with it.
    """
    registry = kc_graph._registry()
    crosswalk = kc_graph._crosswalk()
    return {
        "kc_registry": len(registry),
        "kc_crosswalk": len(crosswalk),
        "qmatrix": len(kc_graph._qmatrix()),
        # Concepts the registry names that the crosswalk cannot measure. A
        # count is not enough on its own: the crosswalk is regenerated from the
        # question bank while the registry is authored, so they drift a concept
        # at a time. Each one that drifts out reports the prior forever and
        # never unlocks its children — the whole-file outage, at retail.
        "kc_unmapped": sum(1 for kc in registry if kc not in crosswalk),
    }


def lattice_is_degraded() -> bool:
    """True when the lattice cannot measure a single concept.

    The crosswalk is the one that silently changes every number rather than
    emptying the graph outright: without the registry there are no nodes at
    all, which is loud, but without the crosswalk there are 63 nodes all
    quietly reporting the prior.
    """
    return lattice_data_health()["kc_crosswalk"] == 0
