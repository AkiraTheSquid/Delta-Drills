#!/usr/bin/env python3
"""Structural audit of the two graphs the app actually serves.

Four families of finding, one ratchet baseline (`graph_structure_baseline.json`):

registry / atoms — sanity of the KC lattice (`lessons/kc_registry.json`) and of
    the atom graph the BKT engine loads (`arena_drillable_v1.json`, NOT the v4
    file — bkt_mastery.GRAPH_PATH is the authority): unknown ids, self-loops,
    cycles, duplicate edges, encompassing weights the runtime would silently
    drop (propagation_weight <= 0) or amplify (> 1).

edge-missing — a drill's solution uses a symbol whose owning KC is EARLIER in
    registry order but NOT an ancestor on the prerequisite lattice. The order
    audit (`audit_solution_prereqs.py`) is blind to this by design: it ranks by
    registry order, which is a linear extension of the lattice, so a symbol can
    pass "taught earlier" while the graph records no path at all — the
    dependency exists in the content but not in the edges. Each finding names a
    candidate edge, aggregated per (kc -> owner) pair, not per drill.

same-move — two drills on the SAME rung of the SAME concept whose solutions
    are the same program after normalization (bound names canonicalized,
    constants reduced to their type, docstrings dropped). Two such drills look
    like variety to the selector and are one exercise to the learner; the
    ladder promotes on the repeat. Constants ARE normalized on purpose:
    `dim=0` vs `dim=1` is the same move.

difficulty — mean difficulty_score inverts across the rung ladder for one
    concept by more than TOLERANCE (a later rung scoring EASIER than an earlier
    one). Informational pressure, not proof; the tolerance absorbs small-n
    noise.

Usage
-----
    python3 scripts/audit_graph_structure.py             # full report
    python3 scripts/audit_graph_structure.py --summary
    python3 scripts/audit_graph_structure.py --new       # vs baseline
    python3 scripts/audit_graph_structure.py --write-baseline
Exit 1 when anything is reported (with --new, when anything is NEW).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_solution_prereqs import (  # noqa: E402
    SELF_DEFINED, declaring_kcs, question_symbols, owner_of)
from audit_lesson_syntax import lesson_order  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "Local_Deployed_Shared"
REGISTRY = SHARED / "lessons" / "kc_registry.json"
QMATRIX = SHARED / "lessons" / "qmatrix_tags.json"
QUESTIONS = SHARED / "questions.json"
STRUCTURED = SHARED / "lessons" / "lessons_structured.json"
QFULL = ROOT / "This-Directory-Only" / "questions_full.json"
ATOM_GRAPH = (ROOT / "This-Directory-Only" / "backend" / "app" / "data"
              / "concept_graphs" / "arena_drillable_v1.json")
BASELINE = Path(__file__).resolve().parent / "graph_structure_baseline.json"

RUNGS = ("faded", "guided", "applied", "solo", "integrated", "independent")
TOLERANCE = 5.0  # difficulty_score points a later rung may sit BELOW an earlier one


def _cycle(nodes, edges_of) -> list[str] | None:
    """One cycle as a node list, or None. Iterative colour DFS."""
    WHITE, GREY, BLACK = 0, 1, 2
    colour = {n: WHITE for n in nodes}
    for root in nodes:
        if colour[root] != WHITE:
            continue
        stack = [(root, iter(edges_of(root)))]
        colour[root], path = GREY, [root]
        while stack:
            node, it = stack[-1]
            for nxt in it:
                if colour.get(nxt, BLACK) == GREY:
                    return path[path.index(nxt):] + [nxt]
                if colour.get(nxt) == WHITE:
                    colour[nxt] = GREY
                    path.append(nxt)
                    stack.append((nxt, iter(edges_of(nxt))))
                    break
            else:
                stack.pop()
                path.pop()
                colour[node] = BLACK
    return None


def check_registry(findings):
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    ids = [kc["id"] for kc in reg["kcs"]]
    rank = {k: i for i, k in enumerate(ids)}
    prereqs = {kc["id"]: list(kc.get("prereqs") or []) for kc in reg["kcs"]}
    for kc, ps in prereqs.items():
        for p in ps:
            if p not in rank:
                findings.append({"key": f"registry|unknown-prereq|{kc}|{p}",
                                 "detail": f"{kc} lists prereq {p!r}, which is not a KC"})
            elif rank[p] >= rank[kc]:
                findings.append({"key": f"registry|order|{kc}|{p}",
                                 "detail": f"{kc} appears before its prerequisite {p}"})
            if p == kc:
                findings.append({"key": f"registry|self-loop|{kc}",
                                 "detail": f"{kc} is its own prerequisite"})
    cyc = _cycle(ids, lambda k: prereqs.get(k, ()))
    if cyc:
        findings.append({"key": "registry|cycle|" + ">".join(cyc),
                         "detail": "prerequisite cycle: " + " -> ".join(cyc)})
    return prereqs, rank


def check_atom_graph(findings):
    g = json.loads(ATOM_GRAPH.read_text(encoding="utf-8"))
    ids = {c["id"] for c in g["concepts"]}
    seen_pairs = set()
    succ = defaultdict(list)
    for e in g["prerequisite_edges"]:
        pre, dep = e["prerequisite_id"], e["dependent_id"]
        pair = (pre, dep)
        tag = f"{pre}->{dep}"
        if pre not in ids or dep not in ids:
            findings.append({"key": f"atoms|dangling|{tag}",
                             "detail": f"edge {tag} references an unknown atom"})
            continue
        if pre == dep:
            findings.append({"key": f"atoms|self-loop|{pre}",
                             "detail": f"atom {pre} depends on itself"})
        if pair in seen_pairs:
            findings.append({"key": f"atoms|duplicate|{tag}",
                             "detail": f"edge {tag} appears more than once"})
        seen_pairs.add(pair)
        succ[pre].append(dep)
        if e.get("is_encompassing"):
            w = float(e.get("propagation_weight", 0.0))
            if not 0.0 < w <= 1.0:
                findings.append({
                    "key": f"atoms|bad-weight|{tag}",
                    "detail": f"encompassing {tag} propagation_weight={w} — "
                              "the runtime silently drops <=0 and amplifies >1"})
    cyc = _cycle(sorted(ids), lambda a: succ.get(a, ()))
    if cyc:
        findings.append({"key": "atoms|cycle|" + ">".join(cyc[:6]),
                         "detail": "atom prerequisite cycle: " + " -> ".join(cyc)})


def _ancestors(prereqs: dict[str, list[str]]) -> dict[str, set[str]]:
    memo: dict[str, set[str]] = {}

    def walk(kc: str, trail: frozenset = frozenset()) -> set[str]:
        if kc in memo:
            return memo[kc]
        if kc in trail:                     # cycle: reported by check_registry
            return set()
        out: set[str] = set()
        for p in prereqs.get(kc, ()):  # noqa: B023
            out.add(p)
            out |= walk(p, trail | {kc})
        memo[kc] = out
        return out

    for kc in prereqs:
        walk(kc)
    return memo


def check_implied_edges(findings, prereqs, rank):
    """Order-only dependencies: taught earlier, but no lattice path."""
    declared, kc_of_page = declaring_kcs()
    order = lesson_order(kc_of_page)
    ancestors = _ancestors(prereqs)
    qmatrix = json.loads(QMATRIX.read_text(encoding="utf-8"))
    bank = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    questions = bank if isinstance(bank, list) else bank.get("questions", bank)
    pairs: dict[tuple[str, str], dict] = {}
    for q in questions:
        qid = q.get("id")
        targets = (qmatrix.get(str(qid)) or {}).get("target_kcs") or []
        ranked = [(order[k], k) for k in targets if k in order]
        if not ranked:
            continue
        my_rank, my_kc = min(ranked)
        for surface in ("solution", "starter"):
            for sym in question_symbols(q, surface):
                if sym in SELF_DEFINED or sym.startswith("syntax."):
                    continue
                owner = owner_of(sym, declared, order)
                if (owner is None or owner == my_kc
                        or order.get(owner, 10**6) > my_rank  # 'late': other audit's
                        or owner in ancestors.get(my_kc, ())):
                    continue
                slot = pairs.setdefault((my_kc, owner), {"qids": set(), "syms": set()})
                slot["qids"].add(qid)
                slot["syms"].add(sym)
    for (kc, owner), ev in sorted(pairs.items()):
        findings.append({
            "key": f"edge-missing|{kc}|{owner}",
            "detail": (f"{kc}: {len(ev['qids'])} drill(s) use "
                       f"{', '.join(sorted(ev['syms'])[:4])} from {owner}, which is "
                       f"earlier in course order but NOT an ancestor on the lattice "
                       f"(qids {sorted(ev['qids'])[:6]}) — candidate prereq edge")})


class _Canon(ast.NodeTransformer):
    """Bound names -> n0,n1,...; constants -> their type. API names survive."""

    def __init__(self, bound: set[str]):
        self.bound, self.map = bound, {}

    def _name(self, name: str) -> str:
        if name not in self.bound:
            return name
        return self.map.setdefault(name, f"n{len(self.map)}")

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id=self._name(node.id), ctx=node.ctx), node)

    def visit_arg(self, node):
        node.arg = self._name(node.arg)
        return node

    def visit_Constant(self, node):
        # Numbers, bools and None are incidental (dim=0 vs dim=1 is the same
        # move); STRINGS are not — an einops pattern like 'c h w -> h (c w)'
        # IS the exercise, and collapsing it reported different rearrangements
        # as duplicates (q347 vs q391, caught on first run).
        if isinstance(node.value, (str, bytes)):
            return node
        return ast.copy_location(ast.Constant(value=type(node.value).__name__), node)

    def visit_FunctionDef(self, node):
        node.name = self._name(node.name)
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)):
            node.body = node.body[1:] or [ast.Pass()]
        return node


def _move_hash(code: str) -> str | None:
    if not code.strip():
        # a missing answer is unknown, not 'the empty program' —
        # ast.parse('') succeeds, and two unknowns are not a dupe
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    bound = {n.id for n in ast.walk(tree)
             if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    bound |= {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    bound |= {n.name for n in ast.walk(tree)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for alias in n.names:
                bound.discard(alias.asname or alias.name.split(".")[0])
    canon = _Canon(bound).visit(tree)
    return hashlib.sha1(ast.dump(canon, annotate_fields=False).encode()).hexdigest()


def check_same_move(findings):
    bank = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    questions = bank if isinstance(bank, list) else bank.get("questions", bank)
    answer = {q["id"]: q.get("answer_code") or "" for q in questions}
    data = json.loads(STRUCTURED.read_text(encoding="utf-8"))
    for lesson in data["lessons"]:
        for kp in lesson["kps"]:
            for rung in RUNGS:
                items = kp.get(f"{rung}_items") or []
                ids = [it["question_id"] if isinstance(it, dict) else it
                       for it in items]
                by_hash: dict[str, int] = {}
                for qid in ids:
                    h = _move_hash(answer.get(qid, ""))
                    if h is None:
                        continue
                    if h in by_hash:
                        a, b = sorted((by_hash[h], qid))
                        findings.append({
                            "key": f"same-move|{kp['kc']}|{rung}|{a}-{b}",
                            "detail": (f"{kp['kc']} {rung}: q{a} and q{b} are the "
                                       "same program after normalization — one "
                                       "exercise wearing two ids")})
                    else:
                        by_hash[h] = qid


def check_difficulty(findings):
    diff = {q["id"]: q.get("difficulty_score")
            for q in json.loads(QFULL.read_text(encoding="utf-8"))}
    data = json.loads(STRUCTURED.read_text(encoding="utf-8"))
    for lesson in data["lessons"]:
        for kp in lesson["kps"]:
            means = []
            for rung in RUNGS:
                items = kp.get(f"{rung}_items") or []
                ids = [it["question_id"] if isinstance(it, dict) else it
                       for it in items]
                vals = [diff[q] for q in ids if isinstance(diff.get(q), (int, float))]
                if vals:
                    means.append((rung, sum(vals) / len(vals), len(vals)))
            for (r1, m1, n1), (r2, m2, n2) in zip(means, means[1:]):
                if m2 < m1 - TOLERANCE:
                    findings.append({
                        "key": f"difficulty|{kp['kc']}|{r1}>{r2}",
                        "detail": (f"{kp['kc']}: {r2} (mean {m2:.0f}, n={n2}) sits "
                                   f"{m1 - m2:.0f} below {r1} (mean {m1:.0f}, n={n1})"
                                   " — later rung is easier")})


def find() -> list[dict]:
    findings: list[dict] = []
    prereqs, rank = check_registry(findings)
    check_atom_graph(findings)
    check_implied_edges(findings, prereqs, rank)
    check_same_move(findings)
    check_difficulty(findings)
    return findings


def load_baseline() -> set[str] | None:
    """None (file absent) and empty set are different answers — see the prereq
    audit's load_baseline for why the ratchet must not treat them alike."""
    if not BASELINE.exists():
        return None
    return set(json.loads(BASELINE.read_text(encoding="utf-8")).get("known") or [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--new", action="store_true")
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args()

    findings = find()
    if args.write_baseline:
        BASELINE.write_text(json.dumps({
            "_": "Known graph-structure findings. watch.py fails on anything "
                 "NOT listed here. Shrink it; re-record only with a reason.",
            "count": len(findings),
            "known": sorted(f["key"] for f in findings),
        }, indent=1) + "\n", encoding="utf-8")
        print(f"baseline written: {len(findings)} finding(s)")
        return 0

    known = load_baseline() or set()
    new = [f for f in findings if f["key"] not in known]
    shown = new if args.new else findings
    if not args.summary:
        for f in sorted(shown, key=lambda f: f["key"]):
            print(f["detail"])
    kinds = defaultdict(int)
    for f in findings:
        kinds[f["key"].split("|", 1)[0]] += 1
    print(f"{len(findings)} finding(s) ({dict(kinds)}); {len(new)} new vs baseline",
          file=sys.stderr)
    return 1 if (new if args.new else findings) else 0


if __name__ == "__main__":
    sys.exit(main())
