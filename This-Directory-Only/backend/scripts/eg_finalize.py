#!/usr/bin/env python3
"""eg_finalize.py — build the final encompassing graph from unified per-edge
judgments (advanced_id / simpler_id / encompassing_fraction).

Steps:
  1. Load v2 baseline + all /tmp/eg_packets/decisions_*.json.
  2. Validate each decision's advanced_id/simpler_id are the edge's two atoms.
  3. Orient every edge: prerequisite_id = simpler_id, dependent_id = advanced_id
     (direction now set by judgment, not the noisy relation labels).
  4. Dedupe parallel edges (same unordered pair): pick direction by
     confidence-weighted vote; keep the max agreeing fraction; merge rationales.
  5. is_encompassing = fraction >= FLOOR; propagation_weight = fraction.
  6. DAG-check BOTH the prereq graph (simpler->advanced) and the encompassing
     credit graph (advanced->simpler). Abort on cycle.
  7. (--write) emit arena_iter5_v3_encompassing.json + reload via loader.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
SRC = BACKEND / "app" / "data" / "concept_graphs" / "arena_iter5_v2.json"
DST = BACKEND / "app" / "data" / "concept_graphs" / "arena_iter5_v3_encompassing.json"
DEC = sorted(Path("/tmp/eg_packets").glob("decisions_*.json"))
FLOOR = 0.05


def has_cycle(nodes, adj):
    color = {n: 0 for n in nodes}
    path = []

    def dfs(u):
        color[u] = 1
        path.append(u)
        for v in adj.get(u, []):
            if color[v] == 1:
                return path[path.index(v):] + [v]
            if color[v] == 0:
                c = dfs(v)
                if c:
                    return c
        path.pop()
        color[u] = 2
        return None

    for n in nodes:
        if color[n] == 0:
            c = dfs(n)
            if c:
                return c
    return None


def main() -> None:
    write = "--write" in sys.argv
    g = json.loads(SRC.read_text(encoding="utf-8"))
    edges = g["prerequisite_edges"]
    concept_ids = {c["id"] for c in g["concepts"]}

    dec = {}
    for f in DEC:
        for d in json.loads(f.read_text(encoding="utf-8")):
            dec[d["edge_index"]] = d
    missing = [i for i in range(len(edges)) if i not in dec]
    if missing:
        raise SystemExit(f"missing decisions for edges: {missing[:15]}")

    # Validate + orient each raw edge by judgment.
    oriented = []  # (simpler, advanced, fraction, confidence, rationale)
    for i, e in enumerate(edges):
        d = dec[i]
        ends = {e["prerequisite_id"], e["dependent_id"]}
        adv, sim = d.get("advanced_id"), d.get("simpler_id")
        if {adv, sim} != ends:
            raise SystemExit(f"edge {i}: advanced/simpler {adv,sim} != endpoints {ends}")
        frac = max(0.0, min(1.0, float(d["encompassing_fraction"])))
        oriented.append((sim, adv, frac, float(d.get("confidence", 0.5)), e.get("rationale", "")))

    # Dedupe by unordered pair: confidence-weighted direction vote; max agreeing frac.
    bypair = defaultdict(list)
    for rec in oriented:
        bypair[frozenset((rec[0], rec[1]))].append(rec)

    final_edges = []
    merged = 0
    for pair, recs in bypair.items():
        if len(recs) > 1:
            merged += len(recs) - 1
        # vote on direction (sim, adv) weighted by confidence
        votes = defaultdict(float)
        for sim, adv, frac, conf, rat in recs:
            votes[(sim, adv)] += conf
        sim, adv = max(votes, key=votes.get)
        # max fraction among recs agreeing with chosen direction
        agree = [r for r in recs if (r[0], r[1]) == (sim, adv)]
        frac = max(r[2] for r in agree)
        conf = max(r[3] for r in agree)
        rats = " | ".join(dict.fromkeys(r[4] for r in recs if r[4]))
        is_enc = frac >= FLOOR
        final_edges.append({
            "prerequisite_id": sim,             # simpler / learn-first
            "dependent_id": adv,                # advanced / builds-on
            "weight": 1.0,
            "confidence": round(conf, 3),
            "is_hard_gate": True,
            "is_encompassing": is_enc,
            "propagation_weight": round(frac, 3) if is_enc else 0.0,
            "rationale": rats,
        })

    # DAG checks
    prereq_adj = defaultdict(list)   # simpler -> advanced
    credit_adj = defaultdict(list)   # advanced -> simpler (encompassing only)
    for e in final_edges:
        prereq_adj[e["prerequisite_id"]].append(e["dependent_id"])
        if e["is_encompassing"]:
            credit_adj[e["dependent_id"]].append(e["prerequisite_id"])
    pc = has_cycle(concept_ids, prereq_adj)
    cc = has_cycle(concept_ids, credit_adj)

    # Report
    enc = [e for e in final_edges if e["is_encompassing"]]
    hist = {"0": 0, "0.05-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for e in final_edges:
        w = e["propagation_weight"]
        if w < FLOOR: hist["0"] += 1
        elif w < 0.2: hist["0.05-0.2"] += 1
        elif w < 0.4: hist["0.2-0.4"] += 1
        elif w < 0.6: hist["0.4-0.6"] += 1
        elif w < 0.8: hist["0.6-0.8"] += 1
        else: hist["0.8-1.0"] += 1
    print(f"raw edges {len(edges)} -> deduped {len(final_edges)} (merged {merged} parallel)")
    print(f"encompassing (frac>={FLOOR}): {len(enc)}/{len(final_edges)} ({100*len(enc)/len(final_edges):.0f}%)")
    if enc:
        print(f"mean nonzero weight: {sum(e['propagation_weight'] for e in enc)/len(enc):.3f}")
    print("weight histogram:")
    for k, v in hist.items():
        print(f"  {k:10s} {v:3d}  {'#'*round(v/3)}")
    print(f"prereq DAG cycle:  {pc}")
    print(f"credit DAG cycle:  {cc}")
    if pc or cc:
        raise SystemExit("CYCLE — aborting before write")

    if write:
        g["prerequisite_edges"] = final_edges
        g["curriculum_id"] = "arena-iter5-v3-encompassing-chap0"
        g["version"] = "0.3.0"
        g["title"] = "ARENA Chapter 0 (iter-5 v3, encompassing graph)"
        g["description"] = (
            g.get("description", "").split(" | ")[0] +
            f" | v3: direction set by per-edge judgment, {merged} parallel edges "
            f"deduped to {len(final_edges)}, {len(enc)} encompassing (continuous "
            "fraction, dependent->prerequisite trickle-down)."
        )
        DST.write_text(json.dumps(g, indent=1), encoding="utf-8")
        from app.concept_graph import load_curriculum_graph
        cg = load_curriculum_graph(str(DST))
        ne = sum(1 for e in cg.prerequisite_edges if e.is_encompassing)
        print(f"WROTE {DST.name}: loads OK, {len(cg.concepts)} concepts, "
              f"{len(cg.prerequisite_edges)} edges, {ne} encompassing")


if __name__ == "__main__":
    main()
