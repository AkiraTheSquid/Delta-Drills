"""kc_report.py — the lattice state the knowledge graph draws.

Split out of `kc_graph.py` on 2026-09-19 when that file crossed Modulario's
700-LOC line; `kc_graph.kc_report` still exists and forwards here, so the
API (`/api/practice/kc-lattice`, the tests) is unchanged. One function, and
it is a READER: every predicate it reports is computed by the same code that
gates practice (`kc_is_learned`, `kc_is_unlocked`, `kc_stage`, `frontier`,
`select_next_kc`), never re-derived here.
"""

from __future__ import annotations

from app import kc_graph, kc_prefs


def kc_report(user_state, eligible=None) -> dict:
    """Full lattice state for the API — one row per KC, plus the selection the
    queue will actually make. This is what lets the knowledge graph draw the
    system's real state instead of a decorative model: every field the graph
    colours by is computed here, by the same code that gates practice.
    """
    reg = kc_graph.registry()
    descendants, depth = kc_graph.closure()
    by_kc = kc_graph.questions_by_kc()
    order = {kc: i for i, kc in enumerate(kc_graph.frontier(user_state))}
    # `eligible` makes the reported next_kc the one the QUEUE will reach, not
    # the frontier head it would like to reach. Without it the graph can ring a
    # concept whose questions are all spent while practice serves the next one
    # along — a highlight that promises something the app then does not do.
    next_kc = kc_graph.select_next_kc(user_state, eligible=eligible)

    rows = {}
    for kc, node in reg.items():
        m, covered, tier = kc_graph.kc_mastery(user_state, kc)
        # Same predicate the practice gate uses, exhaustion credit included —
        # a node the queue treats as cleared must not draw as still-frontier.
        learned = kc_graph.kc_is_learned(user_state, kc)
        unlocked = kc_graph.kc_is_unlocked(user_state, kc)
        ladder_est = kc_graph.kc_estimate(user_state, kc)
        # The concept's OWN graded record, alongside the crosswalk mastery.
        #
        # These answer different questions and the graph has only ever had the
        # first. `mastery` is a BKT posterior over the ATOMS a KC's questions
        # exercise, and the crosswalk that joins concepts to atoms separates
        # only 20 of 63 — for the other 43 the number shown is the topic's,
        # which is why a learner with real drill history still saw bubbles that
        # claimed nothing had been measured about them.
        #
        # The ladder record has no such gap: `record_ladder_outcome` writes one
        # row per graded attempt against every KC the question tags, for all 63.
        # It is a smaller claim than a posterior — k correct out of n, on this
        # concept, recently — but it is this concept's, always, and it is the
        # same quantity the practice topbar draws and the rung gate promotes on.
        # Shipping it here is what lets the graph and the practice screen agree.
        rows[kc] = {
            "title": node["title"],
            "lesson": node["lesson"],
            "topic": node["topic"],
            "prereqs": node["prereqs"],
            "mastery": round(m, 4),
            "covered_w": round(covered, 3),
            "tier": tier,
            "evidenced": covered >= kc_graph.MIN_COVERED_W,
            "state": ("disabled" if kc_prefs.is_disabled(user_state, kc)
                      else "learned" if learned
                      else "frontier" if unlocked else "locked"),
            # The learner's own controls for this concept (graph Settings tab).
            "pref": kc_prefs.pref_row(user_state, kc),
            "coreness": descendants.get(kc, 0),
            "depth": depth.get(kc, 0),
            "n_questions": len(by_kc.get(kc, ())),
            "frontier_rank": order.get(kc),
            "ladder_stage": kc_graph.kc_stage(user_state, kc),
            "ladder_estimate": ladder_est,
        }

    return {
        "learned_threshold": kc_graph.LEARNED_THRESHOLD,
        "next_kc": next_kc,
        "frontier": kc_graph.frontier(user_state),
        "kcs": rows,
    }
