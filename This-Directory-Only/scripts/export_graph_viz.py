#!/usr/bin/env python3
"""Export the canonical concept graph to a lightweight viz JSON for the
front-end Cytoscape.js embed on the How It Works page.

Reads the same graph the backend uses (arena_iter5_v3_encompassing.json) and
writes nodes + edges (plus a family colour grouping) to
Local_Deployed_Shared/concept-graph/graph-viz.json.

Run: python3 This-Directory-Only/scripts/export_graph_viz.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "This-Directory-Only" / "backend" / "app" / "data" / "concept_graphs" / "arena_iter5_v3_encompassing.json"
OUT = ROOT / "Local_Deployed_Shared" / "concept-graph" / "graph-viz.json"

# Coarse families for legible colour grouping. The precise per-node `topic`
# is kept on each node (shown in the tooltip); families just drive colour.
# Edit freely — anything unmapped falls back to "Other".
FAMILY = {
    # Fundamentals: tensors, einops, numerical building blocks
    "Tensor Mechanics": "Fundamentals", "Custom Tensor": "Fundamentals",
    "Tensor Utils": "Fundamentals", "Einops Advanced": "Fundamentals",
    "Numerical Modules": "Fundamentals", "Misc Cleanup": "Fundamentals",
    "Pytorch Modules": "Fundamentals",
    # CNNs / vision
    "CNNs": "CNNs", "Cnn Extras": "CNNs", "Cnn Deep": "CNNs",
    "Geometry Cnn": "CNNs", "Resnet Modules": "CNNs",
    # Ray tracing
    "Ray Tracing": "Ray Tracing",
    # Backprop & autograd
    "Backprop": "Backprop & Autograd", "Backprop Driver": "Backprop & Autograd",
    "Backward Fns": "Backprop & Autograd", "Autograd Internals": "Backprop & Autograd",
    "Autograd Pt2": "Backprop & Autograd", "Autograd Pt3": "Backprop & Autograd",
    # Optimization & training loops
    "Optimization": "Optimization & Training", "Optimizer Internals": "Optimization & Training",
    "Adam Trainer": "Optimization & Training", "Training Loop": "Optimization & Training",
    "Hparam Config": "Optimization & Training", "Logging Instr": "Optimization & Training",
    # Generative models
    "VAEs and GANs": "Generative", "Vae Gan": "Generative",
    "Generative": "Generative", "Dcgan Final": "Generative",
    # Distributed
    "Distributed": "Distributed",
}


def main() -> None:
    g = json.loads(GRAPH.read_text())

    edges = [
        {
            "source": e["prerequisite_id"],   # prereq -> dependent
            "target": e["dependent_id"],
            "enc": bool(e.get("is_encompassing")),
        }
        for e in g["prerequisite_edges"]
    ]

    # Only render skills that actually have a mapped dependency — isolated atoms
    # (no edge yet) just produce a grid of orphan dots. Honest caption on the
    # page notes that edge coverage is still growing.
    connected = set()
    for e in edges:
        connected.add(e["source"])
        connected.add(e["target"])

    nodes = []
    for c in g["concepts"]:
        if c["id"] not in connected:
            continue
        topic = c.get("topic", "")
        nodes.append({
            "id": c["id"],
            "label": c.get("title", c["id"]),
            "topic": topic,
            "family": FAMILY.get(topic, "Other"),
            "tier": c.get("tier", ""),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"nodes": nodes, "edges": edges}, indent=0))
    fams = sorted({n["family"] for n in nodes})
    total = len(g["concepts"])
    print(f"wrote {OUT.relative_to(ROOT)}: {len(nodes)}/{total} connected nodes, {len(edges)} edges")
    print(f"families ({len(fams)}): {fams}")


if __name__ == "__main__":
    main()
