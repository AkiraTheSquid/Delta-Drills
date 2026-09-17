#!/usr/bin/env python3
"""eg_validate.py — validate the encompassing graph v3.

Two layers:
  A. STRUCTURAL — schema loads, no self-loops, weight/flag consistency, no
     parallel pairs left, both prereq + credit graphs acyclic, concepts intact.
  B. BEHAVIORAL — a prototype FIRe propagation (the future EG2) is run on real
     scenarios to prove trickle-down behaves: credit flows advanced->simpler,
     decays with depth, never exceeds direct mastery, atoms with no encompassing
     path get zero, and it terminates.

Exits non-zero if any assertion fails. Prints concrete trickle-down examples.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from app.concept_graph import load_curriculum_graph  # noqa

V3 = BACKEND / "app" / "data" / "concept_graphs" / "arena_iter5_v3_encompassing.json"
FLOOR = 0.05
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail and not cond else ''}")


# ---------- A. STRUCTURAL ----------
print("A. STRUCTURAL")
g = load_curriculum_graph(str(V3))  # raises if schema/consistency invalid
ids = {c.id for c in g.concepts}
E = g.prerequisite_edges
check("schema loads via CurriculumGraph", True)
check("393 concepts preserved", len(g.concepts) == 393, f"got {len(g.concepts)}")
check("no self-loops", all(e.prerequisite_id != e.dependent_id for e in E))
check("endpoints valid", all(e.prerequisite_id in ids and e.dependent_id in ids for e in E))
check("flag⟺weight≥floor",
      all((e.is_encompassing and e.propagation_weight >= FLOOR) or
          (not e.is_encompassing and e.propagation_weight == 0.0) for e in E))
check("weights in (0,1]", all(0 < e.propagation_weight <= 1.0 for e in E if e.is_encompassing))
pairs = defaultdict(int)
for e in E:
    pairs[frozenset((e.prerequisite_id, e.dependent_id))] += 1
check("no parallel pairs after dedup", all(v == 1 for v in pairs.values()),
      f"{sum(v>1 for v in pairs.values())} dup pairs")

# acyclicity (credit graph advanced->simpler)
credit = defaultdict(list)
for e in E:
    if e.is_encompassing:
        credit[e.dependent_id].append((e.prerequisite_id, e.propagation_weight))

def acyclic(adj):
    color = {}
    def dfs(u):
        color[u] = 1
        for v, _ in adj.get(u, []):
            if color.get(v) == 1 or (color.get(v) != 2 and dfs(v)):
                return True
        color[u] = 2
        return False
    return not any(color.get(n) != 2 and dfs(n) for n in list(adj))
check("credit graph acyclic", acyclic(credit))

# ---------- B. BEHAVIORAL (prototype FIRe propagation) ----------
print("\nB. BEHAVIORAL — prototype trickle-down propagation")

def propagate(mastered_id, mastery=1.0, depth_decay=1.0):
    """Credit trickles advanced->simpler along encompassing edges, multi-hop.
    credit[v] combines via noisy-OR so it never exceeds 1.0. Processed by BFS
    over the (acyclic) credit graph from the mastered atom."""
    credit = {mastered_id: mastery}
    # relax in topological-ish order via repeated BFS layers (DAG → terminates)
    frontier = [(mastered_id, mastery)]
    while frontier:
        nxt = []
        for node, cred in frontier:
            for child, w in credit.get(node, []) if False else credit_edges(node):
                add = cred * w * depth_decay
                prev = credit.get(child, 0.0)
                new = 1 - (1 - prev) * (1 - add)   # noisy-OR combine
                if new - prev > 1e-6:
                    credit[child] = new
                    nxt.append((child, add))
        frontier = nxt
    return credit

def credit_edges(node):
    return credit.get(node, [])

title = {c.id: c.title for c in g.concepts}
# pick top encompassers by out-degree in the credit graph
top = sorted(credit, key=lambda n: -len(credit[n]))[:3]

all_credits = []
for src in top:
    cr = propagate(src)
    cr.pop(src)
    all_credits.append((src, cr))
    ranked = sorted(cr.items(), key=lambda kv: -kv[1])[:6]
    print(f"\n  master {title[src]!r} (1.0) → trickles to {len(cr)} atoms:")
    for cid, c in ranked:
        print(f"      {c:.3f}  {title.get(cid, cid)}")

# behavioral assertions
check("trickle-down credit never exceeds 1.0",
      all(c <= 1.0 + 1e-9 for _, cr in all_credits for c in cr.values()))
check("trickle-down credit strictly < direct mastery (1.0)",
      all(c < 1.0 for _, cr in all_credits for c in cr.values()))
# an atom with NO encompassing in-path from src must get 0 credit
non_enc_targets = ids - {e.prerequisite_id for e in E if e.is_encompassing}
src0, cr0 = all_credits[0]
check("atoms with no encompassing-in-edge get zero credit",
      all(cr0.get(t, 0.0) == 0.0 for t in non_enc_targets))
check("propagation terminates (returned)", True)
check("top encompasser actually trickles to >0 atoms", all(len(cr) > 0 for _, cr in all_credits))

# ---------- summary ----------
print(f"\nRESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
