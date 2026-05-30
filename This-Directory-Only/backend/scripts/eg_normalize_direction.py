#!/usr/bin/env python3
"""eg_normalize_direction.py — normalize edge direction in the iter-5 graph.

The iter-5 graph stores edges as semantic triples `prerequisite_id [relation]
dependent_id` (e.g. "Matmul [uses] broadcasting"). That reads correctly as a
relation, but the *prerequisite_id* slot is only "learn-first / simpler" for
some relations:

  relation         prereq slot holds   → action to make prereq = simpler
  ---------------  ------------------   --------------------------------
  uses             the USER (advanced)  FLIP
  refines          the REFINEMENT (adv) FLIP
  part-of          the PART (simpler)   keep
  is-a             the INSTANCE (simpler, treat concept as advanced) keep
  alternative-to   lateral (no order)   keep (not a true prereq)

After normalization, `prerequisite_id` is uniformly the simpler / learn-first
atom and `dependent_id` the more advanced one — so "encompassing credit flows
dependent -> prerequisite" holds for every edge. Rationale text is preserved
verbatim (it still records the original subject-relation-object), so the flip
is fully traceable.

Usage:
  python3 scripts/eg_normalize_direction.py            # report only
  python3 scripts/eg_normalize_direction.py --write     # write normalized graph
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
SRC = BACKEND / "app" / "data" / "concept_graphs" / "arena_iter5_v2.json"
DST = BACKEND / "app" / "data" / "concept_graphs" / "arena_iter5_v3_normalized.json"

FLIP_RELATIONS = {"uses", "refines"}
_PREFIX_RE = re.compile(r"\s*\[([^\]]+)\]")


def prefix_of(rationale: str) -> str:
    m = _PREFIX_RE.match(rationale or "")
    return m.group(1) if m else "<none>"


def has_cycle(nodes, adj):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}
    path = []

    def dfs(u):
        color[u] = GRAY
        path.append(u)
        for v in adj.get(u, []):
            if color[v] == GRAY:
                return path[path.index(v):] + [v]
            if color[v] == WHITE:
                c = dfs(v)
                if c:
                    return c
        path.pop()
        color[u] = BLACK
        return None

    for n in nodes:
        if color[n] == WHITE:
            c = dfs(n)
            if c:
                return c
    return None


def main() -> None:
    write = "--write" in sys.argv
    g = json.loads(SRC.read_text(encoding="utf-8"))
    edges = g["prerequisite_edges"]

    flips = Counter()
    for e in edges:
        rel = prefix_of(e.get("rationale", ""))
        if rel in FLIP_RELATIONS:
            e["prerequisite_id"], e["dependent_id"] = e["dependent_id"], e["prerequisite_id"]
            flips[rel] += 1

    nodes = {c["id"] for c in g["concepts"]}
    adj = defaultdict(list)
    for e in edges:
        adj[e["prerequisite_id"]].append(e["dependent_id"])  # simpler -> advanced
    cycle = has_cycle(nodes, adj)

    print(f"flipped: {dict(flips)}  (total {sum(flips.values())}/{len(edges)})")
    print(f"prereq-graph (simpler->advanced) cycle: {cycle}")
    if cycle:
        print(f"  cycle len {len(cycle)}: {' -> '.join(cycle[:8])}{'...' if len(cycle)>8 else ''}")
        raise SystemExit("CYCLE after normalization — inspect before writing")

    if write:
        g["curriculum_id"] = "arena-iter5-v3-normalized-chap0"
        g["version"] = "0.3.0-normalized"
        g["title"] = "ARENA Chapter 0 (iter-5 v3, direction-normalized)"
        g["description"] = (
            g.get("description", "") +
            f" | v3-normalized: flipped {sum(flips.values())} edges "
            "(uses/refines) so prerequisite_id is uniformly the simpler atom."
        )
        DST.write_text(json.dumps(g, indent=1), encoding="utf-8")
        print(f"WROTE {DST.name}")


if __name__ == "__main__":
    main()
