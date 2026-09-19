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
from app.adaptive import AttemptRecord, UserPracticeState  # noqa: E402

NOW = "2026-08-31T00:00:00+00:00"

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

# Last attempt missed, at `partial` -> the floor is `partial` (faded retired
# 2026-09-11), so the rung stays; the two old faded rows lift onto it.
check("a miss steps the rung down from where it happened, floored at partial",
      kc_graph.kc_stage(st, SOME_KC) == "partial", kc_graph.kc_stage(st, SOME_KC))
st.kc_ladder[SOME_KC]["attempts"][-1]["stage"] = "solo"
check("a miss at solo steps down to partial",
      kc_graph.kc_stage(st, SOME_KC) == "partial", kc_graph.kc_stage(st, SOME_KC))

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

check("the drill floor is partial — faded is retired",
      kc_graph.DRILL_FLOOR == "partial" and kc_graph.LIVE_STAGES == ("worked", "partial", "solo"))
check("_step_down respects the floor",
      kc_graph._step_down("partial", floor=kc_graph.DRILL_FLOOR) == "partial")
check("_step_down still steps where there is room",
      kc_graph._step_down("solo", floor=kc_graph.DRILL_FLOOR) == "partial")
check("a row filed at the retired rung lifts onto the floor",
      kc_graph._floored("faded") == "partial" and kc_graph._floored(None) == "partial"
      and kc_graph._floored("solo") == "solo")

for lowest in ("partial", "faded"):
    st = fresh_state()
    st.kc_ladder[SOME_KC] = {
        "worked_seen": 1,
        "attempts": [{"correct": False, "stage": lowest, "ts": "2026-07-30T00:00:00+00:00"}],
    }
    check(f"a miss on the lowest drill rung stays on the floor (filed at {lowest})",
          kc_graph.kc_stage(st, SOME_KC) == "partial", kc_graph.kc_stage(st, SOME_KC))

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
      kc_graph.kc_stage(st, SOME_KC) == "partial", kc_graph.kc_stage(st, SOME_KC))

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
#
# Promotion off `partial` needs TWO things since the solo_progress gate
# (167b921f): the window/streak arithmetic AND six distinct correct solo drills,
# two of them from the rung's hardest third. `_solo_rows` builds rows that
# satisfy the gate from the KC's real drills, hardest first.
from app.questions import get_question_by_id  # noqa: E402


def _solo_rows(kc, n, correct=True, ts="2026-07-30T00:00:00+00:00"):
    ids = kc_graph.questions_at_stage(kc_graph.questions_for_kc(kc), "partial")
    by_hard = sorted(ids, key=lambda q: -get_question_by_id(q).difficulty_score)
    return [{"correct": correct, "stage": "partial", "ts": ts,
             "question_id": by_hard[i % len(by_hard)]} for i in range(n)]


st = fresh_state()
st.kc_ladder[SOME_KC] = {"worked_seen": 1, "attempts": _solo_rows(SOME_KC, 4)}
row = kc_graph.kc_report(st)["kcs"][SOME_KC]
check("the reported stage equals kc_stage",
      row["ladder_stage"] == kc_graph.kc_stage(st, SOME_KC), row["ladder_stage"])
check("the reported estimate equals kc_estimate",
      row["ladder_estimate"] == kc_graph.kc_estimate(st, SOME_KC), repr(row["ladder_estimate"]))
check("four straight correct clear the window bar but the band gate holds",
      row["ladder_stage"] == "partial" and row["ladder_estimate"]["promote_lo"] >= kc_graph.PROMOTE_LO["partial"]
      and not row["ladder_estimate"]["solo_progress"]["ready"], row["ladder_stage"])
st.kc_ladder[SOME_KC] = {"worked_seen": 1, "attempts": _solo_rows(SOME_KC, 6)}
row = kc_graph.kc_report(st)["kcs"][SOME_KC]
check("six distinct correct, two of them hard, promote to integrated",
      row["ladder_stage"] == "solo" and row["ladder_estimate"]["solo_progress"]["ready"],
      f"{row['ladder_stage']} {row['ladder_estimate']['solo_progress']}")
check("evidence in one KC does not leak into another",
      all(r["ladder_estimate"]["n"] == 0 for kc, r in kc_graph.kc_report(st)["kcs"].items()
          if kc != SOME_KC))

print("\n--- a run of correct answers earns a rung, whatever the window says ---")
# Seth's real record on torch.tensor-model, 2026-08-04: eleven misses from the
# days when the drills themselves were wrong, then seven right out of nine and
# five in a row. Over a 20-attempt window that is 7/20, a Wilson lower bound of
# 0.18, and a `faded` bar of 0.34 — so the rung had not moved in twenty
# questions and would not have moved for another dozen. Meanwhile a SINGLE miss
# demotes immediately. The ladder was listening to one recent answer going down
# and to a twenty-question average going up.
POISONED = "FFFFFFFFFFFTTFFTTTTT"


# With the faded rung retired the run is made at `partial` and the rung it
# buys is `solo`, which is also gated on the band ladder — so the correct rows
# carry real question ids (hardest first) and the gate is satisfied by any
# six of them.
_SOLO_IDS = [r["question_id"] for r in _solo_rows(SOME_KC, 40)]


def ladder(seq, stage="partial"):
    st = fresh_state()
    rows, t = [], 0
    for c in seq:
        row = {"correct": c == "T", "stage": stage, "ts": "2026-08-04T00:00:00+00:00"}
        if c == "T" and stage == "partial":
            row["question_id"] = _SOLO_IDS[t]
            t += 1
        rows.append(row)
    st.kc_ladder[SOME_KC] = {"worked_seen": 1, "attempts": rows}
    return kc_graph.kc_stage(st, SOME_KC)


check("a poisoned window no longer pins a learner who is plainly getting it",
      ladder(POISONED) == "solo", ladder(POISONED))
check("three in a row is the bar, two is not",
      (ladder(POISONED[:-2] + "TTT"), ladder("FFFFFFFFFFFTTFFTT")) == ("solo", "partial"),
      repr((ladder(POISONED[:-2] + "TTT"), ladder("FFFFFFFFFFFTTFFTT"))))
# Rows filed at the RETIRED rung: a run made there lands on the floor (the rung
# it would have promoted to), never on the top. 11 misses then 8 correct is
# 8/19, lower bound 0.24, under the 0.51 bar, so the window cannot promote it
# either.
LONG_RUN = "F" * 11 + "T" * 8
check("a run made at the retired faded rung lands on the floor, not the top",
      ladder(LONG_RUN, stage="faded") == "partial", ladder(LONG_RUN, stage="faded"))
check("...and the same run made at partial reaches solo",
      ladder(LONG_RUN) == "solo", ladder(LONG_RUN))
# The window still promotes on its own where it legitimately can — the streak
# rule is additive and must not have replaced it.
check("a clean record still reaches solo on the window alone",
      ladder("T" * 12) == "solo", ladder("T" * 12))
# The immediate demotion is untouched: a streak is worth one rung and one miss
# gives it straight back, which is what keeps the promotion honest.
check("one miss still gives the rung straight back",
      ladder(POISONED + "F") == "partial", ladder(POISONED + "F"))
check("a cold record: three correct clear the streak but not the band gate; six do",
      (ladder("TTT"), ladder("TTTTTT"), ladder("FFFF")) == ("partial", "solo", "partial"),
      repr((ladder("TTT"), ladder("TTTTTT"), ladder("FFFF"))))

print("\n--- an exhausted supported rung repeats; it never falls through to solo ---")
# The other half of the same complaint: on a concept every one of whose drills
# has been served, `select_next_kc` found nothing unserved, dropped the KC
# narrowing entirely, and the difficulty picker handed a `solo` problem to a
# learner sitting on `Worked`. Twenty attempts against nine drills is the
# ordinary state of a concept revisited, not an edge case.
from app import prioritization  # noqa: E402

LADDER_KC = "torch.tensor-model"
POOL = list(kc_graph.questions_for_kc(LADDER_KC))


class _Q:
    def __init__(self, qid, score=50):
        self.id = qid
        self.difficulty_score = score
        # The Solo band walk reads the aim for the drill's subtopic
        # (prioritization.target_difficulty); a stub without one is a
        # subtopic the learner has never touched, which is the cold aim.
        self.subtopic = "test-subtopic"


def _learn_prereqs(state, kc):
    """Put the learner where this fixture claims they are: on `kc`.

    Without this the fixture was a learner sitting on torch.tensor-model's
    faded rung who had never done any python — impossible since the python
    course went in front of it. `frontier` drops a locked KC, so BOTH of
    `narrow_to_next_kc`'s frontier lookups returned None and every case here
    took the last-resort path. Seeding the prerequisite atoms to mastered is
    what a real learner arriving on this concept looks like, and it makes the
    checks below exercise the ORDINARY narrowing rather than the fallback.
    """
    node = kc_graph.registry_node(kc) or {}
    for parent in node.get("prereqs") or []:
        row = kc_graph._crosswalk().get(parent) or {}
        for atom in row.get("atoms") or []:
            state.atom_mastery[atom["a"]] = 1.0
            state.atom_last_ts[atom["a"]] = NOW


def _state_on_rung(seq, learned_prereqs=True):
    st = fresh_state()
    st.kc_ladder[LADDER_KC] = {
        "worked_seen": 1,
        "attempts": [
            {"correct": c == "T", "stage": "partial", "ts": "2026-08-04T00:00:00+00:00"}
            for c in seq
        ],
    }
    if learned_prereqs:
        _learn_prereqs(st, LADDER_KC)
    return st


def _seed_answered(state, qids, missed=()):
    """Put a graded attempt on the record for each question.

    `narrow_to_next_kc` reads ANSWERED, not served, and it derives that from
    `SubtopicState.history` (prioritization.answered_question_ids). Seeding real
    history rather than passing a set keeps the fixture on the same path the
    router takes. `missed` are the ones whose latest attempt was wrong
    (prioritization.missed_question_ids).
    """
    sub = state.get_subtopic_state("Numpy: Core array literacy")
    for qid in sorted(qids):
        ok = qid not in set(missed)
        sub.history.append(AttemptRecord(
            question_id=int(qid),
            subtopic="Numpy: Core array literacy",
            difficulty_score=50,
            grade=100.0 if ok else 0.0,
            correct=ok,
            timestamp=NOW,
        ))


def narrowed_for(seq, served, learned_prereqs=True, answered=None, missed=(),
                 last_served=None, scores=None):
    st = _state_on_rung(seq, learned_prereqs)
    served = set(served)
    # The cases below all mean "the learner has DONE these", so answered
    # defaults to served. Pass `answered` explicitly to describe the other
    # case — drills that were handed over and skipped.
    _seed_answered(st, served if answered is None else set(answered), missed)
    scores = scores or {}
    out, _kc, gap = prioritization.narrow_to_next_kc(
        st, [_Q(i, scores.get(i, 50)) for i in POOL], served=served, last_served=last_served
    )
    return kc_graph.kc_stage(st, LADDER_KC), sorted(q.id for q in out), gap


def servable(seq, served, learned_prereqs=True, answered=None):
    stage, ids, _gap = narrowed_for(seq, served, learned_prereqs, answered)
    return stage, ids


SOLO = set(kc_graph.questions_at_stage(POOL, "solo"))
check("this concept has a solo rung to leak from", bool(SOLO) and len(POOL) > len(SOLO),
      f"{len(POOL)} drills, solo={sorted(SOLO)}")
# "Every drill served" keeps one Solo drill missed on the record: a learner
# whose every Solo drill is answered CORRECTLY has cleared the rung and is
# promoted for it (kc_graph.solo_rung_cleared, checked below), so a stuck
# learner with the whole pool behind them has to have a miss outstanding.
_first_floor = sorted(kc_graph.questions_at_stage(POOL, "partial"))[:1]
for label, served, missed in (("nothing served", [], ()), ("every drill served", POOL, _first_floor)):
    stage, ids, _ = narrowed_for("FFFFFFFFFFFTTFFTT", served, missed=missed)
    check(f"stuck on the floor, {label}: no integrated drill",
          stage == "partial" and not (set(ids) & SOLO), f"stage={stage} servable={ids}")
    stage, ids, _ = narrowed_for(POISONED, served, missed=missed)
    check(f"on the run, {label}: the rung it earned, not the top",
          stage == "partial" and not (set(ids) & SOLO), f"stage={stage} servable={ids}")
# A concept with every drill served serves NOTHING, and says why. That is the
# 2026-08-28 rule ("it should notify the user that they need to make the AI
# create more problems rather than serving up the old problems they have
# already done"), so the contract to check is not "something comes back" — it
# is that the empty list arrives WITH a gap naming the concept and the rung.
# The router turns that into the 409 the learner reads; an empty list and no
# gap would be a silent 404 instead.
for seq in ("", "FFFF", "TTT", POISONED):
    _stage, _ids, _gap = narrowed_for(seq, POOL)
    check(f"a fully-answered concept reports its gap rather than repeating ({seq or 'cold'})",
          not _ids and bool(_gap) and _gap.get("kc") == LADDER_KC
          and _gap.get("stage") == _stage,
          f"served={_ids} gap={_gap}")

# 🔴 SERVED IS NOT ANSWERED (2026-08-31). Every drill handed over and NONE of
# them answered is a learner who skipped, reloaded, or closed the tab — not a
# learner who has finished the concept. Reporting a gap there spends content
# nobody has done, and on `python.values-and-names` (two drills at its lowest
# authored rung, and the only root of the whole course) it 409'd Seth's account
# out of every concept there is. The Skip button's own label promises "nothing
# is recorded"; this is that promise.
for seq in ("", "FFFF", POISONED):
    _stage, _ids, _gap = narrowed_for(seq, POOL, answered=[])
    check(f"every drill served but none answered still serves ({seq or 'cold'})",
          bool(_ids) and _gap is None and not (set(_ids) & SOLO),
          f"stage={_stage} servable={_ids} gap={_gap}")

# And the half-and-half case: the rung is spent only by the drills that were
# actually answered.
_rung_floor = sorted(kc_graph.questions_at_stage(POOL, "partial"))
_stage, _ids, _gap = narrowed_for("FFFF", POOL, answered=_rung_floor[:1])
check("a rung counts only the drills that were answered",
      _stage == "partial" and _ids and _rung_floor[0] not in _ids and _gap is None,
      f"stage={_stage} answered={_rung_floor[:1]} servable={_ids} gap={_gap}")
check("the retired faded rung serves nothing",
      kc_graph.questions_at_stage(POOL, "faded") == [])

# A Skip on a band of ONE drill must not hand that drill back — the on-screen
# guard in question_pick would read the repeat as the concept having run dry.
# The drill on screen gets its own, easiest band (score 30 against the stub's
# 50), so the bottom-up walk lands on a band that IS that one drill. The
# example-bearing drills come first on this rung (kc_graph.with_example_first),
# so the drill put on screen is one of those or it would be filtered out anyway.
_on_screen = kc_graph.with_example_first(_rung_floor)[0]
_stage, _ids, _gap = narrowed_for("FFFF", [_on_screen], answered=[], last_served=_on_screen,
                                  scores={_on_screen: 30})
check("the drill on screen is not the only thing offered after a skip",
      _stage == "partial" and _ids and _on_screen not in _ids and _gap is None,
      f"stage={_stage} servable={_ids} gap={_gap}")
_stage, _ids, _gap = narrowed_for("FFFF", [_on_screen], answered=[], last_served=None,
                                  scores={_on_screen: 30})
check("...and with nothing on screen that band is served as usual",
      _stage == "partial" and _ids == [_on_screen] and _gap is None,
      f"stage={_stage} servable={_ids} gap={_gap}")

# 🔴 A MISS IS OWED A RETRY (2026-09-11). Promotion off Solo needs six DISTINCT
# correct answers (solo_progress), counted on the latest attempt per drill, and
# a miss spends the drill for the unseen-first order — so a six-drill bank with
# one miss topped out at five and Integrated was unreachable (Seth,
# torch.aggregations, q497). Once nothing unseen is left, the missed drills come
# back, with no gap: the drill exists and the learner has seen its answer.
_missed = _rung_floor[:1]
_stage, _ids, _gap = narrowed_for("FFFF", POOL, missed=_missed)
check("a missed solo drill comes back once nothing unseen is left",
      _stage == "partial" and _ids == sorted(_missed) and _gap is None,
      f"stage={_stage} servable={_ids} gap={_gap}")
_stage, _ids, _gap = narrowed_for("FFFF", POOL, answered=_rung_floor[:2], missed=_missed)
check("...but not while an unseen drill remains",
      _stage == "partial" and _ids and _missed[0] not in _ids and _gap is None,
      f"stage={_stage} servable={_ids} gap={_gap}")
_stage, _ids, _gap = narrowed_for("FFFF", POOL, missed=POOL)
check("every drill missed: the whole rung comes back for a retake, still no leak",
      _stage == "partial" and set(_ids) == set(_rung_floor) and not (set(_ids) & SOLO) and _gap is None,
      f"stage={_stage} servable={_ids} gap={_gap}")

# 🔴 AN AIDED CORRECT ANSWER IS OWED A RETAKE TOO (2026-09-18). The example
# schedule shows a worked example on Solo positions 0/2/5/9 and after every
# miss; a drill answered correctly behind one is spent for the unseen-first
# order but can never count toward the six distinct UNAIDED successes the gate
# wants (solo_progress.successful_questions). Replayed on Seth's state, every
# Solo-rung 409 in the course was this shape. Once nothing unseen is left the
# aided ones come back, unaided.
def _aided_state(seq_ok, aided_qids, answered):
    st = _state_on_rung("", True)
    row = st.kc_ladder[LADDER_KC]
    row["attempts"] = [
        {"correct": True, "stage": "partial", "ts": "2026-08-04T00:00:00+00:00",
         "question_id": int(q), "example": q in set(aided_qids)}
        for q in answered
    ]
    _seed_answered(st, answered)
    out, _kc, gap = prioritization.narrow_to_next_kc(
        st, [_Q(i, 50) for i in POOL], served=set(answered), last_served=None
    )
    return kc_graph.kc_stage(st, LADDER_KC), sorted(q.id for q in out), gap

# Twelve of the fifteen behind an example: three unaided successes is a streak,
# but not the six distinct the gate wants, so the learner is still on Solo.
_aided = _rung_floor[:-3]
_stage, _ids, _gap = _aided_state(True, _aided, _rung_floor)
check("solo drills answered correctly behind an example come back once nothing unseen is left",
      _stage == "partial" and _ids == sorted(_aided) and _gap is None,
      f"stage={_stage} servable={_ids} gap={_gap}")
_stage, _ids, _gap = _aided_state(True, _aided, _rung_floor[:3])
check("...but not while an unseen solo drill remains",
      _stage == "partial" and _ids and not (set(_ids) & set(_rung_floor[:3])) and _gap is None,
      f"stage={_stage} servable={_ids} gap={_gap}")

# 🔴 DEMOTED ONTO A SPENT RUNG (2026-09-18). A miss on an Integrated drill
# steps the learner down to Solo; with every Solo drill answered and none owed
# there was nothing to walk down onto and the queue 409'd on a concept whose
# Solo rung the learner had finished (torch.constructors after q648 missed).
# The missed Integrated drill is the one still owed.
_solo_first = sorted(SOLO)[0]
_st = _state_on_rung("TTTTTT", True)
_row = _st.kc_ladder[LADDER_KC]
_row["attempts"] = [
    {"correct": True, "stage": "partial", "ts": "2026-08-04T00:00:00+00:00", "question_id": int(q), "example": False}
    for q in _rung_floor
] + [{"correct": False, "stage": "solo", "ts": "2026-08-04T00:00:01+00:00", "question_id": int(_solo_first), "example": False}]
_seed_answered(_st, _rung_floor + [_solo_first], missed=[_solo_first])
_out, _kc2, _gap = prioritization.narrow_to_next_kc(
    _st, [_Q(i, 50) for i in POOL], served=set(_rung_floor + [_solo_first]), last_served=None
)
_stage = kc_graph.kc_stage(_st, LADDER_KC)
check("demoted onto a spent solo rung: the missed integrated drill comes back, no 409",
      _stage == "partial" and sorted(q.id for q in _out) == [_solo_first] and _gap is None,
      f"stage={_stage} servable={sorted(q.id for q in _out)} gap={_gap}")

# 🔴 A SPENT SOLO RUNG WITH EVERY ANSWER CORRECT (2026-09-18). Miss, retake,
# miss another, retake: the twenty-attempt window reads "struggling", the
# trailing streak is short, nothing is unseen and nothing is owed — the rung
# is finished and the queue 409'd (replay: torch.broadcasting-rules on nine
# Solo drills). Solving every drill on the rung unaided is the evidence the
# rung asks for; it promotes on its own (kc_graph.solo_rung_cleared).
_st = _state_on_rung("TTTTTT", True)
_row = _st.kc_ladder[LADDER_KC]
_poison = []
for q in _rung_floor:
    _poison.append({"correct": False, "stage": "partial", "ts": "2026-08-04T00:00:00+00:00", "question_id": int(q), "example": False})
    _poison.append({"correct": True, "stage": "partial", "ts": "2026-08-04T00:00:01+00:00", "question_id": int(q), "example": False})
_row["attempts"] = _poison[-kc_graph._LADDER_WINDOW:]
_seed_answered(_st, _rung_floor)
check("the poisoned window alone would hold the learner at partial",
      kc_graph._stage_from(kc_graph.kc_estimate(_st, LADDER_KC), _row) == "partial")
_out, _kc2, _gap = prioritization.narrow_to_next_kc(
    _st, [_Q(i, 50) for i in POOL], served=set(_rung_floor), last_served=None
)
_stage = kc_graph.kc_stage(_st, LADDER_KC)
check("every solo drill solved unaided clears the rung: stage solo, integrated drills served, no 409",
      _stage == "solo" and _out and set(q.id for q in _out) <= SOLO and _gap is None,
      f"stage={_stage} servable={sorted(q.id for q in _out)} gap={_gap}")
_row["attempts"][-1] = dict(_row["attempts"][-1], example=True)
check("...not while the last one was answered behind an example",
      kc_graph.kc_stage(_st, LADDER_KC) == "partial")
_row["attempts"][-1] = dict(_row["attempts"][-1], example=False, correct=False)
_seed_answered(_st, _rung_floor, missed=[_rung_floor[-1]])
check("...and never over an outstanding miss",
      kc_graph.kc_stage(_st, LADDER_KC) == "partial")

# 🔴 THE TOP RUNG SPENT WITH A DRILL OWED (2026-09-18). Every Integrated
# drill answered — one behind the entry example, one missed and retaken —
# (the aided one last, so the recency clause cannot call it finished) and
# the learner back on `solo`: nothing unseen anywhere, so the walk-down
# 409'd, though the aided one is still owed an unaided retake.
_solo_ids = sorted(SOLO)
_st = _state_on_rung("TTTTTT", True)
_row = _st.kc_ladder[LADDER_KC]
_row["attempts"] = [
    {"correct": True, "stage": "partial", "ts": "2026-08-04T00:00:00+00:00", "question_id": int(q), "example": False}
    for q in _rung_floor
] + [{"correct": True, "stage": "solo", "ts": "2026-08-04T00:00:01+00:00", "question_id": int(q), "example": q == _solo_ids[-1]}
     for q in _solo_ids]
_seed_answered(_st, _rung_floor + _solo_ids)
_st.get_subtopic_state("Numpy: Core array literacy").served_question_ids.extend(POOL)
_out, _kc2, _gap = prioritization.narrow_to_next_kc(
    _st, [_Q(i, 50) for i in POOL], served=set(POOL), last_served=None
)
_stage = kc_graph.kc_stage(_st, LADDER_KC)
check("integrated rung spent: the drill answered behind the example comes back, no 409",
      _stage == "solo" and sorted(q.id for q in _out) == [_solo_ids[-1]] and _gap is None,
      f"stage={_stage} servable={sorted(q.id for q in _out)} gap={_gap}")
check("...and the concept is not yet learned while it is owed",
      not kc_graph.kc_evidence_exhausted(_st, LADDER_KC))
_row["attempts"][-1] = dict(_row["attempts"][-1], example=False)
_row["attempts"][-2] = dict(_row["attempts"][-2], stage="partial")
check("every servable drill answered correctly and unaided: evidence exhausted, whatever the tail",
      kc_graph.kc_evidence_exhausted(_st, LADDER_KC))
# Walk-down: an Integrated learner with a missed Solo drill on the record is
# served that retake, not a 409.
_seed_answered(_st, _rung_floor + _solo_ids, missed=_rung_floor[:1])
_out, _kc2, _gap = prioritization.narrow_to_next_kc(
    _st, [_Q(i, 50) for i in POOL], served=set(POOL), last_served=None
)
check("walk-down from a spent integrated rung serves the owed solo drill",
      sorted(q.id for q in _out) == _rung_floor[:1] and _gap and _gap.get("served_from") == "partial",
      f"servable={sorted(q.id for q in _out)} gap={_gap}")

# The frontier can also miss entirely — `frontier` drops a KC that is
# `kc_is_learned`, and `kc_evidence_exhausted` makes that true of any concept
# whose drills are all served. That used to return the pool UNNARROWED, which
# skipped every rung check below it and let a solo drill reach a faded learner.
# Reproduced here by leaving the prerequisites unlearned, which takes the same
# last-resort path.
_stage, _ids = servable("FFFFFFFFFFFTTFFTT", [], learned_prereqs=False)
check("a concept off the frontier is still narrowed to the learner's rung",
      _stage == "partial" and _ids and not (set(_ids) & SOLO),
      f"stage={_stage} servable={_ids}")

# 🔴 A RETAKE IS NOT HANDED STRAIGHT BACK (2026-09-18). With the retakes above
# wired in, replay of Seth's state served the same missed drill up to
# seventeen times in a row: miss → aided retake → aided-correct is owed →
# unaided retake → miss. `attempt_history.RETAKE_COOLDOWN` keeps the last few
# answered drills out of the retry order while anything else is owed, and
# only while — a cooling drill still comes back when it is the only work left.
from app import attempt_history  # noqa: E402


def _touch(state, qid, ts, correct=False, subtopic="Numpy: Applied patterns and advanced"):
    state.get_subtopic_state(subtopic).history.append(AttemptRecord(
        question_id=int(qid), subtopic=subtopic, difficulty_score=50,
        grade=100.0 if correct else 0.0, correct=correct, timestamp=ts))


_st = _state_on_rung("FFFF", True)
_owed2 = _rung_floor[:2]
_seed_answered(_st, POOL, missed=_owed2)
# The second owed drill is the most recent answer, across subtopics.
_touch(_st, _owed2[1], "2026-09-01T00:00:00+00:00")
check("recent answers are read across subtopics, newest last",
      _owed2[1] in attempt_history.recent_question_ids(_st, 1)
      and _owed2[0] not in attempt_history.recent_question_ids(_st, 1),
      f"recent={attempt_history.recent_question_ids(_st, 1)}")
_out, _kc2, _gap = prioritization.narrow_to_next_kc(
    _st, [_Q(i, 50) for i in POOL], served=set(POOL), last_served=None)
check("of two owed drills, the one just answered waits its cooldown",
      sorted(q.id for q in _out) == [_owed2[0]] and _gap is None,
      f"servable={sorted(q.id for q in _out)} gap={_gap}")

_st = _state_on_rung("FFFF", True)
_seed_answered(_st, POOL, missed=_rung_floor[:1])
_touch(_st, _rung_floor[0], "2026-09-01T00:00:00+00:00")
_out, _kc2, _gap = prioritization.narrow_to_next_kc(
    _st, [_Q(i, 50) for i in POOL], served=set(POOL), last_served=None)
check("the only owed drill comes back even inside its cooldown — never a 409",
      sorted(q.id for q in _out) == _rung_floor[:1] and _gap is None,
      f"servable={sorted(q.id for q in _out)} gap={_gap}")

_st = _state_on_rung("FFFF", True)
_seed_answered(_st, POOL, missed=_rung_floor[:4])
for _k, _qid in enumerate(_rung_floor[:4]):
    _touch(_st, _qid, f"2026-09-01T00:00:0{_k}+00:00")
_out, _kc2, _gap = prioritization.narrow_to_next_kc(
    _st, [_Q(i, 50) for i in POOL], served=set(POOL), last_served=None)
check(f"the cooldown is {attempt_history.RETAKE_COOLDOWN} answers wide",
      attempt_history.RETAKE_COOLDOWN == 3 and sorted(q.id for q in _out) == _rung_floor[:1] and _gap is None,
      f"servable={sorted(q.id for q in _out)} gap={_gap}")

# 🔴 A SPENT RUNG ON A LEARNED CONCEPT IS NOT A GAP (2026-09-19). Replay of
# Seth's state: once the frontier lived in other subtopics, weakest-first kept
# landing in ar-02, whose resident concept (cnn.stride-views) was learned and
# spent 500 picks earlier — and every one of those picks wrote a gap record
# and pinned a "ran out" strip on the unrelated drill it served instead.
_st = _state_on_rung("TTTTTT", True)
_seed_answered(_st, POOL)
for _atom in (kc_graph._crosswalk().get(LADDER_KC) or {}).get("atoms") or []:
    _st.atom_mastery[_atom["a"]] = 1.0
    _st.atom_last_ts[_atom["a"]] = NOW
check("fixture: the concept is learned", kc_graph.kc_is_learned(_st, LADDER_KC))
_out, _kc2, _gap = prioritization.narrow_to_next_kc(
    _st, [_Q(i, 50) for i in POOL], served=set(POOL), last_served=None)
check("a spent rung on a learned concept reports no gap and is served as review",
      _gap is None and _kc2 == LADDER_KC and _out and not (set(q.id for q in _out) - set(POOL)),
      f"gap={_gap} kc={_kc2} servable={sorted(q.id for q in _out)}")
_st2 = _state_on_rung("TTTTTT", True)
_seed_answered(_st2, POOL)
_out, _kc2, _gap = prioritization.narrow_to_next_kc(
    _st2, [_Q(i, 50) for i in POOL], served=set(POOL), last_served=None)
check("...while the same spent rung on an UNLEARNED concept still does",
      _gap is not None and _gap.get("kc") == LADDER_KC, f"gap={_gap}")

print()
if fails:
    print(f"FAILED ({len(fails)}): " + ", ".join(fails))
    sys.exit(1)
print("All checks passed.")
