#!/usr/bin/env python3
"""export_display_graph.py — build the graph the MAIN APP renders.

The full concept graph contains a large reference layer of concepts that have
no drillable exercise (extracted from ARENA prose). Those edges are NOT useful
to a student looking at their own skill map, and on the current iter-5 graph
they are ~79% of all edges (the disjoint-vocabulary problem). The app should
render only the structure that relates to things a student actually practices.

This exporter keeps only edges that TOUCH at least one drillable atom (an atom
with >=1 problem_link), drops pure reference-layer (non-drillable -> non-drillable)
edges, and tags each surviving edge for rendering:

    render_kind = "gating"      -> pure prerequisite (is_encompassing == False)
    render_kind = "encompassing"-> encompassing prerequisite (is_encompassing == True)

Plus a per-node `drillable` flag so the renderer can style undrilled nodes
(if any survive as endpoints) differently from practiced atoms.

Once the drillable-atom re-extraction lands, every node is drillable and this
filter becomes a no-op on edges — it then only adds render metadata. Same code
path, no rework.

Usage:
    python scripts/export_display_graph.py [SRC.json] [OUT.json]
Defaults: SRC = active graph, OUT = app/data/concept_graphs/display_graph.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
DEFAULT_SRC = BACKEND / "app" / "data" / "concept_graphs" / "arena_iter5_v4_encompassing.json"
DEFAULT_OUT = BACKEND / "app" / "data" / "concept_graphs" / "display_graph.json"


def build_display_graph(graph: dict) -> dict:
    concepts = graph["concepts"]
    ids = {c["id"] for c in concepts}
    drillable = {
        p["concept_id"]
        for p in graph.get("problem_links", [])
        if p.get("concept_id") in ids
    }

    kept_edges = []
    for e in graph["prerequisite_edges"]:
        pre, dep = e["prerequisite_id"], e["dependent_id"]
        touches_drillable = pre in drillable or dep in drillable
        if not touches_drillable:
            continue  # drop pure reference-layer (N -> N) edges
        kept_edges.append({
            "prerequisite_id": pre,
            "dependent_id": dep,
            "render_kind": "encompassing" if e.get("is_encompassing") else "gating",
            "is_hard_gate": e.get("is_hard_gate", False),
            "weight": e.get("weight", 1.0),
            "propagation_weight": e.get("propagation_weight", 0.0),
        })

    # keep only nodes that appear in a kept edge OR are drillable (so isolated
    # practiced atoms still show as dots — the disconnection is the story)
    endpoint_ids = {e["prerequisite_id"] for e in kept_edges} | {
        e["dependent_id"] for e in kept_edges
    }
    keep_node = lambda cid: cid in drillable or cid in endpoint_ids
    kept_nodes = [
        {
            "id": c["id"],
            "title": c["title"],
            "topic": c.get("topic"),
            "drillable": c["id"] in drillable,
        }
        for c in concepts
        if keep_node(c["id"])
    ]

    return {
        "source": graph.get("curriculum_id"),
        "n_nodes": len(kept_nodes),
        "n_edges": len(kept_edges),
        "n_gating": sum(1 for e in kept_edges if e["render_kind"] == "gating"),
        "n_encompassing": sum(1 for e in kept_edges if e["render_kind"] == "encompassing"),
        "nodes": kept_nodes,
        "edges": kept_edges,
    }


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    graph = json.loads(src.read_text(encoding="utf-8"))
    display = build_display_graph(graph)
    out.write_text(json.dumps(display, indent=1), encoding="utf-8")
    print(f"src: {src.name}")
    print(f"out: {out}")
    print(
        f"display graph: {display['n_nodes']} nodes, {display['n_edges']} edges "
        f"({display['n_gating']} gating + {display['n_encompassing']} encompassing); "
        f"dropped pure reference-layer edges"
    )


if __name__ == "__main__":
    main()
