"""Derive vocab/prereqs.json from vocab/concept_edges.json + vocab/prereq_manual.json.

Concept edges (semantic source of truth) carry one of five kinds, each with a
fixed direction convention:

    A `is-a` B          A is an instance of B
    A `refines` B       A is a specific case of B
    A `uses` B          A invokes / depends on B
    A `part-of` B       A is a structural component of B
    A `alternative-to` B   A and B serve the same need (symmetric)

The prereq derivation maps each concept edge to a directed prereq edge
(A → B in the prereq file means A must be understood before B):

    is-a (A→B)          ⇒ B → A   (category before instance)
    refines (A→B)       ⇒ B → A   (general before specific)
    uses (A→B)          ⇒ B → A   (dependency before dependent)
    part-of (A→B)       ⇒ B → A   (whole before part — ARENA teaches top-down: introduce the system, then drill into its parts)
    alternative-to      ⇒ no prereq induced

Manual edges from prereq_manual.json are added as-is.

Each emitted prereq edge carries a `provenance` list explaining which concept
edge(s) and/or manual entry produced it.

Run from repo root:
    python concept-graph/scripts/derive_prereqs.py
Exit non-zero if the derived prereq graph is not a DAG.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path


KIND_TO_PREREQ_DIRECTION = {
    "is-a": "reverse",
    "refines": "reverse",
    "uses": "reverse",
    "part-of": "reverse",
    "alternative-to": None,
}


def derive(concept_edges: list[dict], manual_edges: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, str], dict] = {}

    def merge(from_id: str, to_id: str, status: str, provenance: str) -> None:
        key = (from_id, to_id)
        if key in by_key:
            existing = by_key[key]
            existing["provenance"].append(provenance)
            if status == "accepted" and existing["status"] != "accepted":
                existing["status"] = "accepted"
        else:
            by_key[key] = {
                "from": from_id,
                "to": to_id,
                "status": status,
                "provenance": [provenance],
            }

    for e in concept_edges:
        kind = e["kind"]
        direction = KIND_TO_PREREQ_DIRECTION.get(kind)
        if direction is None:
            continue
        if direction == "reverse":
            src, dst = e["to"], e["from"]
        else:
            src, dst = e["from"], e["to"]
        prov = f"concept:{kind}({e['from']}→{e['to']})"
        merge(src, dst, e["status"], prov)

    for e in manual_edges:
        prov = f"manual:{e.get('rationale', 'no-rationale')[:80]}"
        merge(e["from"], e["to"], e.get("status", "proposed"), prov)

    return list(by_key.values())


def check_dag(edges: list[dict]) -> list[str]:
    """Return [] if acyclic, else list of nodes in the cyclic component."""
    adj: dict[str, list[str]] = defaultdict(list)
    nodes: set[str] = set()
    for e in edges:
        adj[e["from"]].append(e["to"])
        nodes.add(e["from"])
        nodes.add(e["to"])
    indeg = {n: 0 for n in nodes}
    for u in nodes:
        for v in adj[u]:
            indeg[v] += 1
    q = deque(n for n in nodes if indeg[n] == 0)
    visited = 0
    while q:
        u = q.popleft()
        visited += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if visited == len(nodes):
        return []
    return sorted(n for n in nodes if indeg[n] > 0)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "vocab" / "concept_edges.json").open() as f:
        concept = json.load(f)
    with (repo_root / "vocab" / "prereq_manual.json").open() as f:
        manual = json.load(f)
    with (repo_root / "vocab" / "atoms.json").open() as f:
        vocab = json.load(f)

    known = {a["id"] for a in vocab["atoms"]}

    endpoint_errors = []
    for e in concept["edges"]:
        for k in ("from", "to"):
            if e[k] not in known:
                endpoint_errors.append(f"concept edge {k}='{e[k]}' not in vocab")
    for e in manual["edges"]:
        for k in ("from", "to"):
            if e[k] not in known:
                endpoint_errors.append(f"manual edge {k}='{e[k]}' not in vocab")
    if endpoint_errors:
        for x in endpoint_errors:
            print(f"[ERROR] {x}")
        return 1

    derived = derive(concept["edges"], manual["edges"])
    derived.sort(key=lambda e: (e["from"], e["to"]))

    cycle_nodes = check_dag(derived)
    if cycle_nodes:
        print("[ERROR] derived prereq graph contains a cycle. Nodes involved:")
        for n in cycle_nodes:
            print(f"  {n}")
        out_path = repo_root / "vocab" / "prereqs.json"
        out_path.write_text(json.dumps({
            "schema_version": "0.1",
            "_generated_by": "scripts/derive_prereqs.py (FAILED — cycle in derivation)",
            "edges": derived,
        }, indent=2) + "\n")
        print(f"  partial output written to {out_path} for inspection")
        return 1

    out_path = repo_root / "vocab" / "prereqs.json"
    out_path.write_text(json.dumps({
        "schema_version": "0.1",
        "_generated_by": "scripts/derive_prereqs.py — do not edit by hand. Edit concept_edges.json or prereq_manual.json and re-run.",
        "edges": derived,
    }, indent=2) + "\n")

    concept_count = sum(1 for e in concept["edges"] if KIND_TO_PREREQ_DIRECTION.get(e["kind"]) is not None)
    print(f"concept edges total:    {len(concept['edges'])}")
    print(f"  → induce prereqs:     {concept_count}")
    print(f"  → alternative-to (no prereq): {len(concept['edges']) - concept_count}")
    print(f"manual edges:           {len(manual['edges'])}")
    print(f"derived prereq edges:   {len(derived)}  (after de-dup)")
    print(f"DAG: OK ({len(set(n for e in derived for n in (e['from'], e['to'])))}/{len(known)} atoms touched)")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
