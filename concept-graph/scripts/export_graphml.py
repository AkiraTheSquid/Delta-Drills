"""Export atoms + edges to yEd-friendly GraphML files, optionally pre-grouped.

By default writes FOUR files at concept-graph/:
  - concept-graph.graphml         flat (no groups), semantic edges
  - prereq-graph.graphml          flat, derived prereqs
  - concept-graph-by-part.graphml grouped by arena_part, semantic edges
  - prereq-graph-by-part.graphml  grouped by arena_part, derived prereqs

Use --group-by={none,part,domain} to control which grouping the *-by-*.graphml
files use (default 'part'); --skip-flat to suppress the un-grouped exports.

Open the .graphml in yEd → Tools → Fit Node to Label → Layout → Hierarchical.
With groups: in the Hierarchical dialog enable Grouping → Layout Grouping →
Recursive (the default usually works).

Run from repo root:
    python concept-graph/scripts/export_graphml.py
    python concept-graph/scripts/export_graphml.py --group-by domain
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape


GRAPHML_OPEN = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xmlns:y="http://www.yworks.com/xml/graphml"
         xmlns:yed="http://www.yworks.com/xml/yed/3"
         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd">
  <key for="node" id="d_id" attr.name="id" attr.type="string"/>
  <key for="node" id="d_label" attr.name="label" attr.type="string"/>
  <key for="node" id="d_definition" attr.name="definition" attr.type="string"/>
  <key for="node" id="d_domain" attr.name="domain" attr.type="string"/>
  <key for="node" id="d_status" attr.name="status" attr.type="string"/>
  <key for="node" id="d_dd_coverage" attr.name="dd_coverage" attr.type="string"/>
  <key for="node" id="d_part" attr.name="arena_part" attr.type="int"/>
  <key for="node" id="d_group" attr.name="group" attr.type="string"/>
  <key for="node" id="d6" yfiles.type="nodegraphics"/>
  <key for="edge" id="d_kind" attr.name="kind" attr.type="string"/>
  <key for="edge" id="d_status" attr.name="status" attr.type="string"/>
  <key for="edge" id="d_evidence" attr.name="evidence" attr.type="string"/>
  <key for="edge" id="d10" yfiles.type="edgegraphics"/>
  <graph id="G" edgedefault="directed">
"""

GRAPHML_CLOSE = """  </graph>
</graphml>
"""

PART_LABELS = {
    0: "(orphan — no exercise tag)",
    1: "Part 1 — Ray Tracing",
    2: "Part 2 — CNNs / ResNets",
    3: "Part 3 — Optimization / DDP",
    4: "Part 4 — Backprop from Scratch",
    5: "Part 5 — VAEs / GANs",
}


def domain_to_color(domain: str) -> str:
    h = int(hashlib.md5(domain.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    r, g, b = colorsys.hsv_to_rgb(h, 0.45, 0.95)
    return f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"


def group_fill_color(key: str) -> str:
    h = int(hashlib.md5(f"group:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    r, g, b = colorsys.hsv_to_rgb(h, 0.18, 0.97)
    return f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"


def kind_to_color(kind: str) -> str:
    return {
        "prereq": "#222222",
        "is-a": "#0B6E4F",
        "refines": "#1565C0",
        "uses": "#222222",
        "part-of": "#6A1B9A",
        "alternative-to": "#AD1457",
    }.get(kind, "#555555")


def kind_to_style(kind: str) -> str:
    return {
        "prereq": "line",
        "is-a": "dashed",
        "refines": "dashed",
        "uses": "line",
        "part-of": "dashed_dotted",
        "alternative-to": "dotted",
    }.get(kind, "line")


def status_to_lineweight(status: str) -> str:
    return "2.0" if status == "accepted" else "1.0"


def derive_part(atom_id: str, exercises_by_atom: dict[str, set[int]]) -> int:
    parts = exercises_by_atom.get(atom_id)
    return min(parts) if parts else 0


def build_atom_to_parts(repo_root: Path) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for p in sorted((repo_root / "exercises").glob("*.json")):
        with p.open() as f:
            ex = json.load(f)
        part = ex.get("arena_part", 0)
        for a in ex.get("atoms", []):
            out.setdefault(a["id"], set()).add(part)
    return out


def _group_key_label(atom: dict, part: int, mode: str) -> tuple[str | None, str]:
    if mode == "none":
        return None, ""
    if mode == "part":
        return f"part-{part}", PART_LABELS.get(part, f"Part {part}")
    if mode == "domain":
        d = atom["domain"]
        return f"domain-{d}", d
    raise ValueError(f"unknown group mode: {mode}")


def node_xml(atom: dict, part: int, group_key: str | None, indent: str = "    ") -> str:
    nid = atom["id"]
    color = domain_to_color(atom["domain"])
    label = escape(atom["label"])
    defn = escape(atom["definition"])
    dom = escape(atom["domain"])
    status = escape(atom["status"])
    cov = escape(atom["dd_coverage"])
    width = max(80.0, min(260.0, 8.0 * len(atom["label"]) + 24.0))
    group_data = f"      <data key=\"d_group\">{escape(group_key)}</data>\n" if group_key else ""
    return f"""{indent}<node id="{escape(nid)}">
{indent}  <data key="d_id">{escape(nid)}</data>
{indent}  <data key="d_label">{label}</data>
{indent}  <data key="d_definition">{defn}</data>
{indent}  <data key="d_domain">{dom}</data>
{indent}  <data key="d_status">{status}</data>
{indent}  <data key="d_dd_coverage">{cov}</data>
{indent}  <data key="d_part">{part}</data>
{group_data}{indent}  <data key="d6">
{indent}    <y:ShapeNode>
{indent}      <y:Geometry height="40.0" width="{width:.1f}" x="0.0" y="0.0"/>
{indent}      <y:Fill color="{color}" transparent="false"/>
{indent}      <y:BorderStyle color="#1a1a1a" raised="false" type="line" width="1.0"/>
{indent}      <y:NodeLabel alignment="center" autoSizePolicy="content" fontFamily="Dialog" fontSize="12" textColor="#000000" hasBackgroundColor="false" hasLineColor="false" horizontalTextPosition="center" iconTextGap="4" modelName="internal" modelPosition="c" verticalTextPosition="bottom" visible="true">{label}</y:NodeLabel>
{indent}      <y:Shape type="roundrectangle"/>
{indent}    </y:ShapeNode>
{indent}  </data>
{indent}</node>
"""


def group_node_open(group_key: str, group_label: str) -> str:
    color = group_fill_color(group_key)
    label = escape(group_label)
    gid = escape(group_key)
    return f"""    <node id="g_{gid}" yfiles.foldertype="group">
      <data key="d_id">g_{gid}</data>
      <data key="d_label">{label}</data>
      <data key="d6">
        <y:ProxyAutoBoundsNode>
          <y:Realizers active="0">
            <y:GroupNode>
              <y:Geometry height="50.0" width="160.0" x="0.0" y="0.0"/>
              <y:Fill color="{color}" transparent="false"/>
              <y:BorderStyle color="#2A2A2A" type="dashed" width="1.5"/>
              <y:NodeLabel alignment="center" autoSizePolicy="content" backgroundColor="#FFFFFF" fontFamily="Dialog" fontSize="14" fontStyle="bold" textColor="#000000" hasLineColor="false" horizontalTextPosition="center" iconTextGap="4" modelName="internal" modelPosition="t" verticalTextPosition="bottom" visible="true">{label}</y:NodeLabel>
              <y:Shape type="roundrectangle"/>
              <y:State closed="false" closedHeight="50.0" closedWidth="100.0" innerGraphDisplayEnabled="false"/>
              <y:Insets bottom="20" bottomF="20.0" left="20" leftF="20.0" right="20" rightF="20.0" top="30" topF="30.0"/>
              <y:BorderInsets bottom="0" bottomF="0.0" left="0" leftF="0.0" right="0" rightF="0.0" top="0" topF="0.0"/>
            </y:GroupNode>
            <y:GroupNode>
              <y:Geometry height="50.0" width="160.0" x="0.0" y="0.0"/>
              <y:Fill color="{color}" transparent="false"/>
              <y:BorderStyle color="#2A2A2A" type="dashed" width="1.5"/>
              <y:NodeLabel alignment="center" autoSizePolicy="content" backgroundColor="#FFFFFF" fontFamily="Dialog" fontSize="14" fontStyle="bold" textColor="#000000" hasLineColor="false" horizontalTextPosition="center" iconTextGap="4" modelName="internal" modelPosition="t" verticalTextPosition="bottom" visible="true">{label}</y:NodeLabel>
              <y:Shape type="roundrectangle"/>
              <y:State closed="true" closedHeight="50.0" closedWidth="100.0" innerGraphDisplayEnabled="false"/>
              <y:Insets bottom="20" bottomF="20.0" left="20" leftF="20.0" right="20" rightF="20.0" top="30" topF="30.0"/>
              <y:BorderInsets bottom="0" bottomF="0.0" left="0" leftF="0.0" right="0" rightF="0.0" top="0" topF="0.0"/>
            </y:GroupNode>
          </y:Realizers>
        </y:ProxyAutoBoundsNode>
      </data>
      <graph edgedefault="directed" id="g_{gid}:">
"""


def group_node_close() -> str:
    return "      </graph>\n    </node>\n"


def edge_xml(idx: int, e: dict, default_kind: str = "prereq") -> str:
    kind = e.get("kind", default_kind)
    status = e.get("status", "proposed")
    color = kind_to_color(kind)
    style = kind_to_style(kind)
    weight = status_to_lineweight(status)
    note = escape(e.get("evidence") or e.get("rationale") or " ; ".join(e.get("provenance", [])) or "")
    return f"""    <edge id="e{idx}" source="{escape(e['from'])}" target="{escape(e['to'])}">
      <data key="d_kind">{escape(kind)}</data>
      <data key="d_status">{escape(status)}</data>
      <data key="d_evidence">{note}</data>
      <data key="d10">
        <y:PolyLineEdge>
          <y:LineStyle color="{color}" type="{style}" width="{weight}"/>
          <y:Arrows source="none" target="standard"/>
        </y:PolyLineEdge>
      </data>
    </edge>
"""


def _write_summary_by_part(repo_root: Path, out_name: str, vocab: dict, edges: list[dict],
                            atom_to_parts: dict):
    """Emit a 6-node summary: one box per arena_part, edges weighted by count."""
    atom_part = {a["id"]: derive_part(a["id"], atom_to_parts) for a in vocab["atoms"]}

    pair_counts: dict[tuple[int, int], int] = {}
    parts_used: set[int] = set()
    for e in edges:
        sp = atom_part.get(e["from"], 0)
        tp = atom_part.get(e["to"], 0)
        parts_used.add(sp)
        parts_used.add(tp)
        if sp == tp:
            continue
        pair_counts[(sp, tp)] = pair_counts.get((sp, tp), 0) + 1

    max_count = max(pair_counts.values(), default=1)

    out_path = repo_root / out_name
    with out_path.open("w") as f:
        f.write(GRAPHML_OPEN)

        for part in sorted(parts_used):
            label = escape(PART_LABELS.get(part, f"Part {part}"))
            color = group_fill_color(f"part-{part}")
            atom_count = sum(1 for v in atom_part.values() if v == part)
            full_label = f"{label}\n({atom_count} atoms)"
            f.write(f"""    <node id="part_{part}">
      <data key="d_id">part_{part}</data>
      <data key="d_label">{label}</data>
      <data key="d6">
        <y:ShapeNode>
          <y:Geometry height="80.0" width="240.0" x="0.0" y="0.0"/>
          <y:Fill color="{color}" transparent="false"/>
          <y:BorderStyle color="#2A2A2A" raised="false" type="line" width="2.0"/>
          <y:NodeLabel alignment="center" autoSizePolicy="content" fontFamily="Dialog" fontSize="18" fontStyle="bold" textColor="#000000" hasBackgroundColor="false" hasLineColor="false" horizontalTextPosition="center" iconTextGap="4" modelName="internal" modelPosition="c" verticalTextPosition="bottom" visible="true">{escape(full_label)}</y:NodeLabel>
          <y:Shape type="roundrectangle"/>
        </y:ShapeNode>
      </data>
    </node>
""")

        for i, ((sp, tp), count) in enumerate(sorted(pair_counts.items())):
            ratio = count / max_count
            width = 1.0 + ratio * 7.0
            f.write(f"""    <edge id="se{i}" source="part_{sp}" target="part_{tp}">
      <data key="d_kind">summary</data>
      <data key="d_status">accepted</data>
      <data key="d_evidence">{count} underlying edges</data>
      <data key="d10">
        <y:PolyLineEdge>
          <y:LineStyle color="#222222" type="line" width="{width:.1f}"/>
          <y:Arrows source="none" target="standard"/>
          <y:EdgeLabel alignment="center" backgroundColor="#FFFFFF" fontFamily="Dialog" fontSize="14" fontStyle="bold" textColor="#000000" hasLineColor="false" preferredPlacement="anywhere" ratio="0.5" visible="true">{count}</y:EdgeLabel>
        </y:PolyLineEdge>
      </data>
    </edge>
""")

        f.write(GRAPHML_CLOSE)

    print(f"wrote {out_path}")
    print(f"  {len(parts_used)} part-nodes, {len(pair_counts)} summary edges (max count = {max_count})")


def _write(repo_root: Path, out_name: str, vocab: dict, edges: list[dict],
           atom_to_parts: dict, default_kind: str, group_mode: str):
    out_path = repo_root / out_name
    with out_path.open("w") as f:
        f.write(GRAPHML_OPEN)

        if group_mode == "none":
            for atom in vocab["atoms"]:
                f.write(node_xml(atom, derive_part(atom["id"], atom_to_parts), group_key=None))
        else:
            buckets: dict[str, tuple[str, list[tuple[dict, int]]]] = {}
            for atom in vocab["atoms"]:
                part = derive_part(atom["id"], atom_to_parts)
                key, label = _group_key_label(atom, part, group_mode)
                if key is None:
                    continue
                buckets.setdefault(key, (label, []))[1].append((atom, part))

            def _sort_key(item):
                key = item[0]
                if group_mode == "part":
                    try:
                        return (int(key.split("-")[1]),)
                    except (IndexError, ValueError):
                        return (999,)
                return (item[1][0],)

            for key, (label, members) in sorted(buckets.items(), key=_sort_key):
                f.write(group_node_open(key, label))
                for atom, part in members:
                    f.write(node_xml(atom, part, group_key=key, indent="        "))
                f.write(group_node_close())

        for i, e in enumerate(edges):
            f.write(edge_xml(i, e, default_kind=default_kind))

        f.write(GRAPHML_CLOSE)

    print(f"wrote {out_path}")
    print(f"  {len(vocab['atoms'])} nodes, {len(edges)} edges, group={group_mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-by", choices=("none", "part", "domain"), default="part",
                        help="grouping for the *-by-*.graphml output files (default: part)")
    parser.add_argument("--skip-flat", action="store_true",
                        help="skip the un-grouped (flat) exports")
    parser.add_argument("--summary", action="store_true",
                        help="also emit the 6-node by-part summary (auto-on when --group-by=part)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "vocab" / "atoms.json").open() as f:
        vocab = json.load(f)
    with (repo_root / "vocab" / "concept_edges.json").open() as f:
        concept = json.load(f)
    with (repo_root / "vocab" / "prereqs.json").open() as f:
        prereqs = json.load(f)
    atom_to_parts = build_atom_to_parts(repo_root)

    if not args.skip_flat:
        _write(repo_root, "concept-graph.graphml", vocab, concept["edges"], atom_to_parts,
               default_kind="uses", group_mode="none")
        _write(repo_root, "prereq-graph.graphml", vocab, prereqs["edges"], atom_to_parts,
               default_kind="prereq", group_mode="none")

    if args.group_by != "none":
        suffix = f"-by-{args.group_by}"
        _write(repo_root, f"concept-graph{suffix}.graphml", vocab, concept["edges"], atom_to_parts,
               default_kind="uses", group_mode=args.group_by)
        _write(repo_root, f"prereq-graph{suffix}.graphml", vocab, prereqs["edges"], atom_to_parts,
               default_kind="prereq", group_mode=args.group_by)

    if args.group_by == "part" or args.summary:
        _write_summary_by_part(repo_root, "concept-groups-summary.graphml",
                               vocab, concept["edges"], atom_to_parts)
        _write_summary_by_part(repo_root, "prereq-groups-summary.graphml",
                               vocab, prereqs["edges"], atom_to_parts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
