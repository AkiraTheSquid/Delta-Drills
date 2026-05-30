#!/usr/bin/env python3
"""eg_build_edge_packets.py — Phase B of the encompassing-graph build.

Reads the iter-5 prerequisite graph and emits per-edge "context packets"
grouped into balanced topic clusters. Each packet carries everything an LLM
classifier (Phase C) needs to decide whether a prerequisite edge is ALSO an
*encompassing* edge (encompassing ⊆ prerequisite) and, if so, its
propagation_weight.

We deliberately do NOT pre-decide is_encompassing here. The rationale-prefix
([uses]/[part-of]/[is-a]/[refines]/[alternative-to]) is passed through as ONE
signal among several; direction and semantics are subtle enough that the
classifier must look at both concepts' descriptions.

Output: /tmp/eg_packets/cluster_{i}.json  (i in 0..N-1)
        /tmp/eg_packets/manifest.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

GRAPH = (
    Path(__file__).resolve().parents[1]
    / "app" / "data" / "concept_graphs" / "arena_iter5_v2.json"
)
OUT = Path("/tmp/eg_packets")
N_CLUSTERS = 5

_PREFIX_RE = re.compile(r"\s*\[([^\]]+)\]")


def prefix_of(rationale: str) -> str:
    m = _PREFIX_RE.match(rationale or "")
    return m.group(1) if m else "<none>"


def main() -> None:
    g = json.loads(GRAPH.read_text(encoding="utf-8"))
    concepts = {c["id"]: c for c in g["concepts"]}
    edges = g["prerequisite_edges"]

    def node_brief(cid: str) -> dict:
        c = concepts[cid]
        return {
            "id": c["id"],
            "title": c["title"],
            "topic": c.get("topic", ""),
            "description": c.get("description", ""),
        }

    packets = []
    for i, e in enumerate(edges):
        packets.append({
            "edge_index": i,
            "prefix": prefix_of(e.get("rationale", "")),
            "rationale": e.get("rationale", ""),
            "weight": e.get("weight", 1.0),
            "confidence": e.get("confidence", 1.0),
            "is_hard_gate": e.get("is_hard_gate", True),
            # prerequisite = lower/needed; dependent = higher/builds-on.
            # Encompassing credit (if any) flows dependent -> prerequisite.
            "prerequisite": node_brief(e["prerequisite_id"]),
            "dependent": node_brief(e["dependent_id"]),
        })

    # Balance clusters into even ~equal-size chunks. Dependent-topics only span
    # a handful of coarse buckets (CNNs alone is 154 edges), so whole-topic
    # bucketing is hopelessly lopsided. Instead sort by (dependent, prerequisite)
    # topic to keep related edges adjacent, then split into N contiguous chunks
    # of roughly equal size. Each packet is self-contained, so a topic spanning
    # a chunk boundary is fine.
    packets.sort(key=lambda p: (p["dependent"]["topic"], p["prerequisite"]["topic"]))
    total = len(packets)
    base, extra = divmod(total, N_CLUSTERS)
    buckets: list[list] = []
    start = 0
    for i in range(N_CLUSTERS):
        size = base + (1 if i < extra else 0)
        buckets.append(packets[start:start + size])
        start += size

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"n_clusters": N_CLUSTERS, "total_edges": len(packets), "clusters": []}
    for i, b in enumerate(buckets):
        path = OUT / f"cluster_{i}.json"
        path.write_text(json.dumps(b, indent=1), encoding="utf-8")
        topics = sorted({p["dependent"]["topic"] for p in b})
        manifest["clusters"].append({"file": str(path), "edges": len(b), "topics": topics})
        print(f"cluster_{i}: {len(b)} edges, {len(topics)} topics")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"total {len(packets)} edges -> {OUT}")


if __name__ == "__main__":
    main()
