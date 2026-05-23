"""Validate vocab/concept_edges.json + vocab/prereq_manual.json (the source-of-truth files).

Checks (hard errors → exit 1):
  - Endpoint resolution: every `from`/`to` must be a known atom id.
  - Self-loops: invalid in either file.
  - Duplicate edges in concept_edges: same (from, to, kind) triple.
  - Duplicate edges in prereq_manual: same (from, to) pair.
  - Concept-graph kind must be one of: is-a, refines, uses, part-of, alternative-to.
  - Concept-graph subgraphs of `is-a`, `refines`, `uses`, `part-of` must each be DAGs
    individually. Cycles within `alternative-to` are allowed (symmetric).

Soft warnings (printed, exit 0):
  - Atoms with no concept-graph edges (might be drillable in isolation but worth a look).
  - Overlap between concept-derived prereqs and manual prereqs (redundant).

Run from repo root:
    python concept-graph/scripts/validate_concept.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path

VALID_KINDS = {"is-a", "refines", "uses", "part-of", "alternative-to"}
DIRECTED_KINDS = VALID_KINDS - {"alternative-to"}


def load(repo_root: Path) -> tuple[dict, dict, dict]:
    with (repo_root / "vocab" / "atoms.json").open() as f:
        vocab = json.load(f)
    with (repo_root / "vocab" / "concept_edges.json").open() as f:
        concept = json.load(f)
    with (repo_root / "vocab" / "prereq_manual.json").open() as f:
        manual = json.load(f)
    return vocab, concept, manual


def find_cycles(edges: list[dict]) -> list[str]:
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
    vocab, concept, manual = load(repo_root)
    known = {a["id"] for a in vocab["atoms"]}

    concept_edges = concept["edges"]
    manual_edges = manual["edges"]

    print(f"concept_edges: {len(concept_edges)}")
    print(f"prereq_manual: {len(manual_edges)}")
    print()

    errors: list[str] = []

    for i, e in enumerate(concept_edges):
        for k in ("from", "to", "kind"):
            if k not in e:
                errors.append(f"concept[{i}] missing field: {k}")
        if "kind" in e and e["kind"] not in VALID_KINDS:
            errors.append(f"concept[{i}] invalid kind: {e['kind']}")
        if "from" in e and e["from"] not in known:
            errors.append(f"concept[{i}] unknown from: {e['from']}")
        if "to" in e and e["to"] not in known:
            errors.append(f"concept[{i}] unknown to: {e['to']}")
        if e.get("from") == e.get("to"):
            errors.append(f"concept[{i}] self-loop: {e.get('from')}")

    seen_concept: dict[tuple, int] = {}
    for i, e in enumerate(concept_edges):
        key = (e.get("from"), e.get("to"), e.get("kind"))
        if key in seen_concept:
            errors.append(f"concept[{i}] duplicate of [{seen_concept[key]}]: {key}")
        else:
            seen_concept[key] = i

    for i, e in enumerate(manual_edges):
        for k in ("from", "to"):
            if k not in e:
                errors.append(f"manual[{i}] missing field: {k}")
        if "from" in e and e["from"] not in known:
            errors.append(f"manual[{i}] unknown from: {e['from']}")
        if "to" in e and e["to"] not in known:
            errors.append(f"manual[{i}] unknown to: {e['to']}")
        if e.get("from") == e.get("to"):
            errors.append(f"manual[{i}] self-loop: {e.get('from')}")

    seen_manual: dict[tuple, int] = {}
    for i, e in enumerate(manual_edges):
        key = (e.get("from"), e.get("to"))
        if key in seen_manual:
            errors.append(f"manual[{i}] duplicate of [{seen_manual[key]}]: {key}")
        else:
            seen_manual[key] = i

    if errors:
        print("HARD ERRORS:")
        for x in errors:
            print(f"  {x}")
        return 1

    for kind in DIRECTED_KINDS:
        sub = [e for e in concept_edges if e["kind"] == kind]
        cyc = find_cycles(sub)
        if cyc:
            print(f"HARD ERROR: cycle in '{kind}' subgraph. Nodes:")
            for n in cyc:
                print(f"  {n}")
            return 1

    touched = set()
    for e in concept_edges:
        touched.add(e["from"])
        touched.add(e["to"])
    isolated = sorted(known - touched)
    if isolated:
        print(f"ATOMS with no concept-graph edges ({len(isolated)}):")
        for a in isolated:
            print(f"  {a}")
        print()

    concept_derived = set()
    for e in concept_edges:
        if e["kind"] == "alternative-to":
            continue
        elif e["kind"] == "part-of":
            concept_derived.add((e["to"], e["from"]))
        else:
            concept_derived.add((e["to"], e["from"]))
    manual_pairs = {(e["from"], e["to"]) for e in manual_edges}
    overlap = concept_derived & manual_pairs
    if overlap:
        print(f"REDUNDANT manual edges (already implied by concept_edges): {len(overlap)}")
        for u, v in sorted(overlap):
            print(f"  {u} -> {v}")
        print()

    print("OK: concept_edges + prereq_manual are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
