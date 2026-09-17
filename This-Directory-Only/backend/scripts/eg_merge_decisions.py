#!/usr/bin/env python3
"""eg_merge_decisions.py — Phase D of the encompassing-graph build.

Merges the per-cluster LLM decisions (/tmp/eg_packets/decisions_*.json) back
onto the iter-5 prerequisite graph, producing the encompassing graph v3. Then
validates:
  - every edge_index decided exactly once
  - schema/consistency (via concept_graph._validate_graph)
  - propagation (dependent->prerequisite) edges form a DAG (no credit cycles)

Usage:
  python3 scripts/eg_merge_decisions.py            # report only, no write
  python3 scripts/eg_merge_decisions.py --write     # write v3 graph
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

SRC = BACKEND / "app" / "data" / "concept_graphs" / "arena_iter5_v2.json"
DST = BACKEND / "app" / "data" / "concept_graphs" / "arena_iter5_v3_encompassing.json"
DECISIONS = sorted(Path("/tmp/eg_packets").glob("decisions_*.json"))


AUDIT = Path("/tmp/eg_packets/audit_verdicts.json")


def load_decisions() -> dict[int, dict]:
    merged: dict[int, dict] = {}
    for f in DECISIONS:
        for d in json.loads(f.read_text(encoding="utf-8")):
            idx = d["edge_index"]
            if idx in merged:
                raise SystemExit(f"Duplicate edge_index {idx} across decision files")
            merged[idx] = d
    # Apply adversarial-audit overrides (Phase D quality gate).
    if AUDIT.exists():
        n = 0
        for v in json.loads(AUDIT.read_text(encoding="utf-8")):
            if v.get("verdict") == "change":
                idx = v["edge_index"]
                merged[idx]["is_encompassing"] = bool(v["is_encompassing"])
                merged[idx]["propagation_weight"] = float(v["propagation_weight"])
                merged[idx]["reasoning"] = "[audited] " + v.get("reasoning", "")
                n += 1
        print(f"applied {n} audit overrides")
    return merged


def has_cycle(nodes: set[str], adj: dict[str, list[str]]) -> list[str] | None:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}
    stack_path: list[str] = []

    def dfs(u: str) -> list[str] | None:
        color[u] = GRAY
        stack_path.append(u)
        for v in adj.get(u, []):
            if color[v] == GRAY:
                return stack_path[stack_path.index(v):] + [v]
            if color[v] == WHITE:
                c = dfs(v)
                if c:
                    return c
        stack_path.pop()
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
    dec = load_decisions()

    missing = [i for i in range(len(edges)) if i not in dec]
    extra = [i for i in dec if i < 0 or i >= len(edges)]
    if missing or extra:
        raise SystemExit(f"Decision coverage broken: missing={missing[:10]} extra={extra[:10]}")

    # Continuous-first: each decision carries `encompassing_fraction` in [0,1].
    # propagation_weight = fraction (source of truth); is_encompassing is a
    # derived convenience flag = fraction >= FLOOR. Sub-floor fractions are
    # treated as pure (non-encompassing) prerequisites -> weight 0.
    FLOOR = 0.05
    enc_count = 0
    weights = []
    low_conf = []
    hist = {"0": 0, "0.05-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for i, e in enumerate(edges):
        d = dec[i]
        frac = float(d["encompassing_fraction"])
        frac = max(0.0, min(1.0, frac))
        is_enc = frac >= FLOOR
        w = round(frac, 3) if is_enc else 0.0
        e["is_encompassing"] = is_enc
        e["propagation_weight"] = w
        if is_enc:
            enc_count += 1
            weights.append(w)
        if float(d.get("confidence", 1.0)) < 0.6:
            low_conf.append(i)
        # histogram on the raw fraction
        if frac < FLOOR: hist["0"] += 1
        elif frac < 0.2: hist["0.05-0.2"] += 1
        elif frac < 0.4: hist["0.2-0.4"] += 1
        elif frac < 0.6: hist["0.4-0.6"] += 1
        elif frac < 0.8: hist["0.6-0.8"] += 1
        else: hist["0.8-1.0"] += 1

    # Credit-flow DAG check: encompassing credit flows dependent -> prerequisite.
    nodes = {c["id"] for c in g["concepts"]}
    prereq_adj: dict[str, list[str]] = defaultdict(list)   # prereq -> dependent (gating)
    credit_adj: dict[str, list[str]] = defaultdict(list)   # dependent -> prereq (propagation)
    for e in edges:
        prereq_adj[e["prerequisite_id"]].append(e["dependent_id"])
        if e.get("is_encompassing"):
            credit_adj[e["dependent_id"]].append(e["prerequisite_id"])
    prereq_cycle = has_cycle(nodes, prereq_adj)
    credit_cycle = has_cycle(nodes, credit_adj)

    # Per-prefix breakdown of what got flagged.
    import re
    pref_re = re.compile(r"\s*\[([^\]]+)\]")
    by_prefix = defaultdict(lambda: [0, 0])  # prefix -> [enc, total]
    for e in edges:
        m = pref_re.match(e.get("rationale", ""))
        p = m.group(1) if m else "<none>"
        by_prefix[p][1] += 1
        if e.get("is_encompassing"):
            by_prefix[p][0] += 1

    print(f"encompassing edges (frac>={FLOOR}): {enc_count}/{len(edges)} ({100*enc_count/len(edges):.0f}%)")
    print(f"mean propagation_weight (of nonzero): {sum(weights)/len(weights):.3f}" if weights else "no enc edges")
    print("fraction histogram:")
    for k, v in hist.items():
        print(f"  {k:10s} {v:3d}  {'#'*round(v/3)}")
    print(f"low-confidence (<0.6) decisions: {len(low_conf)} -> {low_conf[:20]}")
    print("by prefix (encompassing/total):")
    for p, (enc, tot) in sorted(by_prefix.items(), key=lambda kv: -kv[1][1]):
        print(f"  {p:16s} {enc:3d}/{tot:3d}  ({100*enc/tot:.0f}%)")
    print(f"prereq-graph cycle: {prereq_cycle}")
    print(f"credit-flow cycle:  {credit_cycle}")

    if prereq_cycle or credit_cycle:
        raise SystemExit("CYCLE DETECTED — aborting before write")

    if write:
        g["curriculum_id"] = "arena-iter5-v3-encompassing-chap0"
        g["version"] = "0.3.0"
        g["title"] = "ARENA Chapter 0 (iter-5 v3, encompassing layer)"
        g["description"] = (
            g.get("description", "") +
            f" | v3: {enc_count} encompassing edges flagged (subset of "
            f"{len(edges)} prereq edges), dependent->prerequisite trickle-down."
        )
        DST.write_text(json.dumps(g, indent=1), encoding="utf-8")
        # Re-validate through the real loader.
        from app.concept_graph import load_curriculum_graph, encompassing_edges_from  # noqa
        cg = load_curriculum_graph(str(DST))
        n_enc = sum(1 for e in cg.prerequisite_edges if e.is_encompassing)
        print(f"WROTE {DST.name}: loads OK, {len(cg.concepts)} concepts, "
              f"{len(cg.prerequisite_edges)} edges, {n_enc} encompassing")


if __name__ == "__main__":
    main()
