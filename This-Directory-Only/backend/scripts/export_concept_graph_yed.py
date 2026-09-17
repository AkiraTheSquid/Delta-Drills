#!/usr/bin/env python3
"""
Export a curriculum concept graph JSON file to GraphML for yEd.

Usage:
  PYTHONPATH=... python3 export_concept_graph_yed.py \
      --input /path/to/graph.json \
      --output /path/to/graph.graphml
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from app.concept_graph import DEFAULT_GRAPH_PATH, load_curriculum_graph

Y_NS = "http://www.yworks.com/xml/graphml"
ET.register_namespace("", "http://graphml.graphdrawing.org/xmlns")
ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
ET.register_namespace("y", Y_NS)


def _node_color(tier: str, kind: str) -> str:
    if tier == "core" and kind == "knowledge":
        return "#9FD0FF"
    if tier == "core" and kind == "skill":
        return "#7ED6A8"
    if tier == "core" and kind == "strategy":
        return "#FFD37E"
    if tier == "supplemental":
        return "#D8D8D8"
    return "#FFFFFF"


def _edge_color(is_hard_gate: bool) -> str:
    return "#3A7AFE" if is_hard_gate else "#A06CD5"


def build_graphml(input_path: Path) -> ET.ElementTree:
    curriculum = load_curriculum_graph(input_path)

    graphml = ET.Element(
        "graphml",
        {
            "{http://www.w3.org/2001/XMLSchema-instance}schemaLocation": (
                "http://graphml.graphdrawing.org/xmlns "
                "http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd"
            )
        },
    )
    ET.SubElement(graphml, "key", {"id": "d0", "for": "node", "yfiles.type": "nodegraphics"})
    ET.SubElement(graphml, "key", {"id": "d1", "for": "edge", "yfiles.type": "edgegraphics"})
    ET.SubElement(graphml, "key", {"id": "d2", "for": "node", "attr.name": "description", "attr.type": "string"})
    ET.SubElement(graphml, "key", {"id": "d3", "for": "edge", "attr.name": "rationale", "attr.type": "string"})

    graph = ET.SubElement(graphml, "graph", {"edgedefault": "directed", "id": curriculum.curriculum_id})

    for concept in curriculum.concepts:
        node = ET.SubElement(graph, "node", {"id": concept.id})

        ET.SubElement(node, "data", {"key": "d2"}).text = (
            f"{concept.title}\n"
            f"topic={concept.topic}\n"
            f"subtopic={concept.subtopic}\n"
            f"tier={concept.tier}\n"
            f"kind={concept.kind}\n"
            f"tags={', '.join(concept.tags)}\n"
            f"description={concept.description}"
        )

        data = ET.SubElement(node, "data", {"key": "d0"})
        shape_node = ET.SubElement(data, f"{{{Y_NS}}}ShapeNode")
        ET.SubElement(
            shape_node,
            f"{{{Y_NS}}}Geometry",
            {"height": "64.0", "width": "220.0", "x": "0.0", "y": "0.0"},
        )
        ET.SubElement(
            shape_node,
            f"{{{Y_NS}}}Fill",
            {"color": _node_color(concept.tier, concept.kind), "transparent": "false"},
        )
        ET.SubElement(
            shape_node,
            f"{{{Y_NS}}}BorderStyle",
            {"color": "#2E3440", "type": "line", "width": "1.0"},
        )
        ET.SubElement(
            shape_node,
            f"{{{Y_NS}}}NodeLabel",
        ).text = f"{concept.title}\n[{concept.topic} / {concept.subtopic}]\n{concept.tier} · {concept.kind}"
        ET.SubElement(shape_node, f"{{{Y_NS}}}Shape", {"type": "roundrectangle"})

    for idx, edge in enumerate(curriculum.prerequisite_edges, start=1):
        edge_el = ET.SubElement(
            graph,
            "edge",
            {"id": f"e{idx}", "source": edge.prerequisite_id, "target": edge.dependent_id},
        )
        ET.SubElement(
            edge_el,
            "data",
            {"key": "d3"},
        ).text = (
            f"weight={edge.weight}\n"
            f"confidence={edge.confidence}\n"
            f"hard_gate={edge.is_hard_gate}\n"
            f"rationale={edge.rationale}"
        )
        data = ET.SubElement(edge_el, "data", {"key": "d1"})
        poly = ET.SubElement(data, f"{{{Y_NS}}}PolyLineEdge")
        ET.SubElement(
            poly,
            f"{{{Y_NS}}}LineStyle",
            {
                "color": _edge_color(edge.is_hard_gate),
                "type": "line" if edge.is_hard_gate else "dashed",
                "width": "1.5",
            },
        )
        ET.SubElement(poly, f"{{{Y_NS}}}Arrows", {"source": "none", "target": "standard"})
        ET.SubElement(poly, f"{{{Y_NS}}}EdgeLabel").text = (
            f"{'hard' if edge.is_hard_gate else 'soft'}\n"
            f"w={edge.weight:.2f}"
        )

    return ET.ElementTree(graphml)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_GRAPH_PATH, help="Path to concept graph JSON.")
    parser.add_argument("--output", type=Path, help="Output GraphML path. Defaults beside input file.")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve() if args.output else input_path.with_suffix(".graphml")

    tree = build_graphml(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(output_path)


if __name__ == "__main__":
    main()
