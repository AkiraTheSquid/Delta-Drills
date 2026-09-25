#!/usr/bin/env python3
"""Syntax gate (app/symbol_gate.py, 2026-09-17): a drill is held back until
every KC teaching a symbol it uses is learned or read.

Run: .venv/bin/python scripts/test_symbol_gate.py
"""
import os
import sys
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path

os.environ["USER_DATA_DIR"] = tempfile.mkdtemp(prefix="symbol_gate_test_")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import kc_graph, lessons, practice_targets, symbol_gate  # noqa: E402
from app.adaptive import get_user_state  # noqa: E402
from app.prioritization import question_is_unlocked  # noqa: E402
from app.questions import get_every_question  # noqa: E402

fails = []


class _Q:
    def __init__(self, qid):
        self.id = qid


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


st = get_user_state(str(uuid.uuid4()))
index = symbol_gate._index()
check("index loaded", len(index) > 1000, f"{len(index)} drills")

# Find a drill whose index lists an owner OFF its chain (neither ancestor nor
# descendant of the target) — the case the gate exists for.
anc = symbol_gate._ancestors()
case = None
for qid, owners in index.items():
    targets = set(kc_graph.question_kcs(qid))
    if not targets:
        continue
    for kc in owners:
        if kc in targets or targets & anc.get(kc, set()):
            continue
        if any(kc in anc.get(t, set()) for t in targets):
            continue
        case = (qid, targets, kc)
        break
    if case:
        break
check("off-chain case exists in bank", case is not None)
qid, targets, owner = case

learned = set().union(*(anc[t] for t in targets)) | targets
kc_graph.kc_is_learned = lambda s, k: k in learned
practice_targets.effective_exposure = lambda s: {}

check("kc gate passes with chain learned", kc_graph.question_kc_gate(st, qid))
blocked = symbol_gate.blocking_kcs(st, qid)
check("off-chain owner blocks", owner in blocked, f"q{qid} {sorted(targets)} needs {owner}: {blocked.get(owner)}")
check("question_is_unlocked = False", not question_is_unlocked(st, _Q(qid)))

# (a) mastered the owning concept -> served
learned.add(owner)
check("learned owner unblocks", owner not in symbol_gate.blocking_kcs(st, qid))
learned.discard(owner)

# (b) read the owning lesson, not yet mastered -> served
exp = {owner: "t"}
for seg in lessons._kc_segments.get(owner, []):
    exp[f"{owner}#{seg['concept_id']}"] = "t"
practice_targets.effective_exposure = lambda s: exp
check("exposed owner unblocks", owner not in symbol_gate.blocking_kcs(st, qid))
practice_targets.effective_exposure = lambda s: {}

# Downstream owner (audit 'late' debt) never gates: would deadlock.
late = None
for q2, owners in index.items():
    t2 = set(kc_graph.question_kcs(q2))
    for kc in owners:
        if t2 & anc.get(kc, set()):
            late = (q2, t2, kc)
            break
    if late:
        break
if late:
    q2, t2, kc = late
    learned = set().union(*(anc[t] for t in t2)) | t2
    check("downstream owner does not gate", kc not in symbol_gate.blocking_kcs(st, q2), f"q{q2} {kc}")

# Whole course stays walkable: frontier walk, only the current KC's lesson read.
# Every course on, so a standalone course's concepts (course_registry) count too.
from app import course_registry  # noqa: E402
for c in course_registry.COURSES:
    st.course_shares[c["id"]] = 0.4
reg = kc_graph._registry()
by_kc = defaultdict(list)
for q in get_every_question():
    for k in kc_graph.question_kcs(q.id):
        by_kc[k].append(q.id)
learned = set()
while True:
    fr = [k for k in reg if k not in learned and all(p in learned for p in reg[k]["prereqs"])]
    moved = False
    for k in fr:
        exp = {k: "t", **{f"{k}#{s['concept_id']}": "t" for s in lessons._kc_segments.get(k, [])}}
        practice_targets.effective_exposure = lambda s, e=exp: e
        ok = [q for q in by_kc.get(k, []) if question_is_unlocked(st, _Q(q))]
        if ok or not by_kc.get(k):
            learned.add(k)
            moved = True
    if not fr or not moved:
        break
check("course walkable under gate", len(learned) == len(reg), f"{len(learned)}/{len(reg)}")

print(f"\n{len(fails)} failure(s)")
sys.exit(1 if fails else 0)
