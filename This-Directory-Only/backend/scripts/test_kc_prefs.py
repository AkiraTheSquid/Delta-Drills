#!/usr/bin/env python3
"""Per-concept learner preferences (graph Settings tab, 2026-09-06).

Covers: weight clamping and the neutral-row removal in `kc_prefs`, a disabled
concept leaving the frontier and no longer gating its dependents, a weight
reordering the frontier, disabled drills failing the question gate, the
`state`/`pref` fields on `kc_report`, and the save/load round trip.

Run: .venv/bin/python scripts/test_kc_prefs.py
Exits non-zero on any failed assertion. No pytest dependency.
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ["USER_DATA_DIR"] = tempfile.mkdtemp(prefix="kc_prefs_test_")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import kc_graph, kc_prefs  # noqa: E402
from app import adaptive  # noqa: E402
from app.adaptive import UserPracticeState  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


REG = kc_graph._registry()

print("\n--- kc_prefs rows ---")
st = UserPracticeState(user_id="prefs-test")
check("unset weight is 1.0", kc_prefs.weight_for(st, "x") == 1.0)
check("clamp: negative → MIN", kc_prefs.clamp_weight(-3) == kc_prefs.MIN_WEIGHT)
check("clamp: 0 → MIN (off is the flag, never the weight)", kc_prefs.clamp_weight(0) == kc_prefs.MIN_WEIGHT)
check("clamp: huge → MAX", kc_prefs.clamp_weight(99) == kc_prefs.MAX_WEIGHT)
check("clamp: garbage → 1.0", kc_prefs.clamp_weight("nope") == 1.0)
kc_prefs.set_pref(st, "a", weight=1.5)
check("weight stored", st.kc_prefs["a"]["weight"] == 1.5)
kc_prefs.set_pref(st, "a", weight=1.0)
check("neutral row removed", "a" not in st.kc_prefs, repr(st.kc_prefs))
kc_prefs.set_pref(st, "a", enabled=False)
check("disabled → weight 0", kc_prefs.weight_for(st, "a") == 0.0)
check("disabled → is_disabled", kc_prefs.is_disabled(st, "a"))
check("pref_row shape", kc_prefs.pref_row(st, "a") == {"enabled": False, "weight": 1.0})
kc_prefs.set_pref(st, "a", enabled=True)
check("re-enabled at 1.0 → row removed", "a" not in st.kc_prefs)

print("\n--- frontier + gate ---")
st = UserPracticeState(user_id="prefs-frontier")
base = kc_graph.frontier(st)
check("fresh learner has a frontier", len(base) >= 1, repr(base))
head = base[0]
kc_prefs.set_pref(st, head, enabled=False)
after = kc_graph.frontier(st)
check("disabled head leaves the frontier", head not in after)
check("its dependents become the frontier", len(after) >= 1, repr(after))
check("report marks it disabled",
      kc_graph.kc_report(st)["kcs"][head]["state"] == "disabled")
kids = [k for k, n in REG.items() if head in n["prereqs"]]
if kids:
    kid = kids[0]
    others_ok = all(kc_graph.kc_is_learned(st, p) or kc_prefs.is_disabled(st, p)
                    for p in REG[kid]["prereqs"])
    check("disabled prereq no longer blocks its dependent",
          kc_graph.kc_is_unlocked(st, kid) == others_ok, f"kid={kid}")
qs = kc_graph.questions_for_kc(head)
if qs:
    check("drill on a disabled concept fails the gate",
          not kc_graph.question_kc_gate(st, qs[0]))
check("next_kc skips the disabled head", kc_graph.select_next_kc(st) != head)

# Weight: push the last frontier node to the front.
st = UserPracticeState(user_id="prefs-weight")
# The python course is a chain at the bottom: switch nodes off from the root
# until the frontier fans out to two or more.
base = kc_graph.frontier(st)
for _ in range(20):
    if len(base) >= 2 or not base:
        break
    kc_prefs.set_pref(st, base[0], enabled=False)
    base = kc_graph.frontier(st)
check("weight fixture has ≥2 frontier nodes", len(base) >= 2, repr(base))
tail = base[-1]
kc_prefs.set_pref(st, tail, weight=kc_prefs.MAX_WEIGHT)
descendants, depth = kc_graph._closure()
score = lambda k: (descendants.get(k, 0) + 1) * kc_prefs.weight_for(st, k)
expected = max(base, key=lambda k: (score(k), -depth.get(k, 0), k))
check("frontier head is the node with the highest weighted coreness",
      kc_graph.frontier(st)[0] == expected, f"expected={expected} got={kc_graph.frontier(st)[:3]}")
check("max weight raised the tail node's score above the old head",
      score(tail) > (descendants.get(base[0], 0) + 1) * 1.0 or expected == tail,
      f"tail={tail} score={score(tail)}")
kc_prefs.set_pref(st, tail, weight=0.75)
check("0.75 keeps it in the frontier", tail in kc_graph.frontier(st))
check("report carries the pref row",
      kc_graph.kc_report(st)["kcs"][tail]["pref"] == {"enabled": True, "weight": 0.75})

print("\n--- persistence ---")
uid = "prefs-persist"
st = adaptive.get_user_state(uid)
kc_prefs.set_pref(st, head, enabled=False, weight=0.5)
adaptive.save_user_state(uid)
adaptive._user_states.pop(uid, None)
re = adaptive.get_user_state(uid)
check("round trip", re.kc_prefs == {head: {"enabled": False, "weight": 0.5}}, repr(re.kc_prefs))

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("All checks passed.")
