#!/usr/bin/env python3
"""Validation suite for the ladder record reaching the knowledge graph (2026-07-30).

Covers: the read-only ladder accessor (`ladder_view`) versus the write path
(`ladder_row`), the promise that reading a learner's stage or estimate never
creates stored state, and the `ladder_stage` / `ladder_estimate` fields on every
row of `kc_report` — which is what `/api/practice/kc-lattice` returns and what
the graph draws from.

The first group is the one that matters. `kc_report` asks every KC for its stage,
and stage lookup used to run through `ladder_row`, which installs a row as a side
effect. Left alone, a single GET of the lattice would have written 63 empty rows
into the learner's persisted state.

Run: .venv/bin/python scripts/test_kc_ladder_report.py
Exits non-zero on any failed assertion. No pytest dependency.
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ["USER_DATA_DIR"] = tempfile.mkdtemp(prefix="kc_ladder_test_")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import kc_graph  # noqa: E402
from app.adaptive import UserPracticeState  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


def fresh_state():
    return UserPracticeState(user_id="ladder-test")


REG = kc_graph._registry()
SOME_KC = sorted(REG)[0]

print("\n--- ladder_view does not create state ---")

st = fresh_state()
view = kc_graph.ladder_view(st, SOME_KC)
check("view of an absent row returns the empty default",
      view == {"worked_seen": 0, "attempts": []}, repr(view))
check("view left kc_ladder empty", st.kc_ladder == {}, repr(st.kc_ladder))

# The returned default must be detached: a caller that mutates it must not be
# able to half-create a row, and must not corrupt the next caller's default.
view["attempts"].append({"correct": True})
check("mutating the default does not touch kc_ladder", st.kc_ladder == {})
check("the next view is unaffected by that mutation",
      kc_graph.ladder_view(st, SOME_KC)["attempts"] == [])

st = fresh_state()
kc_graph.kc_estimate(st, SOME_KC)
kc_graph.kc_stage(st, SOME_KC)
check("kc_estimate + kc_stage create no rows", st.kc_ladder == {}, repr(st.kc_ladder))

print("\n--- ladder_row still writes, because it is the write path ---")

st = fresh_state()
kc_graph.ladder_row(st, SOME_KC)
check("ladder_row installs the row", SOME_KC in st.kc_ladder, repr(st.kc_ladder))

st = fresh_state()
kc_graph.note_worked_seen(st, SOME_KC)
check("note_worked_seen records the exposure",
      st.kc_ladder.get(SOME_KC, {}).get("worked_seen") == 1,
      repr(st.kc_ladder.get(SOME_KC)))

print("\n--- an existing row still reads correctly ---")

st = fresh_state()
st.kc_ladder[SOME_KC] = {
    "worked_seen": 1,
    "attempts": [
        {"correct": True, "stage": "faded", "ts": "2026-07-30T00:00:00+00:00"},
        {"correct": True, "stage": "faded", "ts": "2026-07-30T00:01:00+00:00"},
        {"correct": False, "stage": "partial", "ts": "2026-07-30T00:02:00+00:00"},
    ],
}
est = kc_graph.kc_estimate(st, SOME_KC)
check("n counts the stored attempts", est["n"] == 3, repr(est))
check("correct counts only the right ones", est["correct"] == 2, repr(est))
check("worked_seen is carried through", est["worked_seen"] == 1, repr(est))
check("the interval brackets the point estimate",
      est["ci"][0] <= est["p"] <= est["ci"][1], repr(est))
check("last_ts is the most recent attempt's",
      est["last_ts"] == "2026-07-30T00:02:00+00:00", repr(est["last_ts"]))

# Last attempt missed, at `partial` -> step down one rung to `faded`.
check("a miss steps the rung down from where it happened",
      kc_graph.kc_stage(st, SOME_KC) == "faded", kc_graph.kc_stage(st, SOME_KC))

print("\n--- a row missing its keys does not crash the read ---")

st = fresh_state()
st.kc_ladder[SOME_KC] = {}  # a row written by an older build
est = kc_graph.kc_estimate(st, SOME_KC)
check("an empty row reads as no evidence", est["n"] == 0 and est["worked_seen"] == 0, repr(est))
check("no attempts means no last_ts", est["last_ts"] is None, repr(est["last_ts"]))
check("an empty row still yields a stage", kc_graph.kc_stage(st, SOME_KC) == "worked")

print("\n--- demotion never lands back on the lesson rung ---")

# `worked` is the teaching page, not a drill. Once a concept has been taught,
# no amount of failing may put that page back in front of the learner: the
# demotion re-derives from `attempts[-1]` on every question, so a single miss
# used to replay the whole lesson before every subsequent question on that KC
# until they happened to answer one correctly.

check("_step_down respects the floor",
      kc_graph._step_down("faded", floor="faded") == "faded")
check("_step_down still steps where there is room",
      kc_graph._step_down("solo", floor="faded") == "partial")

st = fresh_state()
st.kc_ladder[SOME_KC] = {
    "worked_seen": 1,
    "attempts": [{"correct": False, "stage": "faded", "ts": "2026-07-30T00:00:00+00:00"}],
}
check("a miss on the lowest drill rung stays on that rung",
      kc_graph.kc_stage(st, SOME_KC) == "faded", kc_graph.kc_stage(st, SOME_KC))

# The other demotion path: not one miss but a confidently bad record. Wilson
# upper at 1/8 is 0.42, under DEMOTE_HI. The last attempt is CORRECT so the
# miss rule above does not fire and this branch is the one under test.
st = fresh_state()
st.kc_ladder[SOME_KC] = {
    "worked_seen": 1,
    "attempts": (
        [{"correct": False, "stage": "solo", "ts": "2026-07-30T00:00:00+00:00"}] * 7
        + [{"correct": True, "stage": "solo", "ts": "2026-07-30T00:07:00+00:00"}]
    ),
}
est = kc_graph.kc_estimate(st, SOME_KC)
check("the confidently-struggling branch is the one being exercised",
      est["ci"][1] < kc_graph.DEMOTE_HI, f"upper={est['ci'][1]}")
check("a bad record restores full support without re-teaching",
      kc_graph.kc_stage(st, SOME_KC) == "faded", kc_graph.kc_stage(st, SOME_KC))

# The one branch that may still return `worked`: nobody has been taught yet.
st = fresh_state()
st.kc_ladder[SOME_KC] = {
    "worked_seen": 0,
    "attempts": [{"correct": False, "stage": "faded", "ts": "2026-07-30T00:00:00+00:00"}],
}
check("an untaught concept still opens on the lesson",
      kc_graph.kc_stage(st, SOME_KC) == "worked", kc_graph.kc_stage(st, SOME_KC))

# And the property that matters, stated directly: across every shape of record
# a taught concept can hold, the lesson rung is unreachable.
taught_rows = []
for stage in kc_graph.LADDER_STAGES:
    for correct in (True, False):
        for n in (1, 4, 12):
            taught_rows.append([
                {"correct": correct, "stage": stage, "ts": "2026-07-30T00:00:00+00:00"}
            ] * n)
offenders = []
for attempts in taught_rows:
    st = fresh_state()
    st.kc_ladder[SOME_KC] = {"worked_seen": 1, "attempts": attempts}
    if kc_graph.kc_stage(st, SOME_KC) == "worked":
        offenders.append((attempts[0]["stage"], attempts[0]["correct"], len(attempts)))
check(f"no taught record reaches the lesson rung ({len(taught_rows)} shapes)",
      not offenders, repr(offenders))

print("\n--- kc_report carries the ladder for every KC ---")

st = fresh_state()
report = kc_graph.kc_report(st)
rows = report["kcs"]
check("every registry KC has a row", set(rows) == set(REG), f"{len(rows)} rows vs {len(REG)} KCs")
check("every row carries a stage",
      all(r.get("ladder_stage") in kc_graph.LADDER_STAGES for r in rows.values()))
check("every row carries an estimate",
      all(isinstance(r.get("ladder_estimate"), dict) for r in rows.values()))
check("a cold learner is on the first rung everywhere",
      all(r["ladder_stage"] == "worked" for r in rows.values()))
check("building the report wrote nothing to kc_ladder",
      st.kc_ladder == {}, f"{len(st.kc_ladder)} rows written")

# And with real evidence, the reported row matches the direct call — the graph
# and the practice topbar must not be able to disagree.
st = fresh_state()
st.kc_ladder[SOME_KC] = {
    "worked_seen": 1,
    "attempts": [{"correct": True, "stage": "faded", "ts": "2026-07-30T00:00:00+00:00"}] * 4,
}
row = kc_graph.kc_report(st)["kcs"][SOME_KC]
check("the reported stage equals kc_stage",
      row["ladder_stage"] == kc_graph.kc_stage(st, SOME_KC), row["ladder_stage"])
check("the reported estimate equals kc_estimate",
      row["ladder_estimate"] == kc_graph.kc_estimate(st, SOME_KC), repr(row["ladder_estimate"]))
check("four straight correct promote off the scaffold",
      row["ladder_stage"] == "solo", row["ladder_stage"])
check("evidence in one KC does not leak into another",
      all(r["ladder_estimate"]["n"] == 0 for kc, r in kc_graph.kc_report(st)["kcs"].items()
          if kc != SOME_KC))

print()
if fails:
    print(f"FAILED ({len(fails)}): " + ", ".join(fails))
    sys.exit(1)
print("All checks passed.")
