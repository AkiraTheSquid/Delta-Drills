#!/usr/bin/env python3
"""Validation suite for kc_stats — the attempt log's first reader (2026-09-01).

Covers: trailing-run stall detection (including that a demotion RESTARTS the
run and a promotion ENDS one), the maximal low-predicted-p run, per-stage and
per-KC aggregation across several learners, legacy stage normalization on read,
and that flags fire exactly at their floors. Synthetic logs are written through
`attempt_log.AttemptRow`/`append` itself, so a schema drift there breaks here.

Run: .venv/bin/python scripts/test_kc_stats.py
Exits non-zero on any failed assertion. No pytest dependency.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import attempt_log, kc_stats  # noqa: E402

BASE = Path(tempfile.mkdtemp(prefix="kc_stats_test_"))

fails = []


def check(name, cond, detail=""):
    tag = "ok  " if cond else "FAIL"
    print(f"  {tag} {name}" + (f"  {detail}" if (detail and not cond) else ""))
    if not cond:
        fails.append(name)


def put(uid, kc, stage, correct, ts, p=0.5):
    attempt_log.append(
        attempt_log.AttemptRow(
            ts=ts, kind=attempt_log.KIND_ATTEMPT, user_id=uid, kc=kc,
            question_id=1, stage=stage, features={"ability": 0.0},
            predicted_p=p, correct=correct,
        ),
        base_dir=BASE,
    )


T = "2026-09-01T00:00:{:02d}+00:00"

# u1 on kc.a: 2 faded, promoted to solo, then demoted and held on faded x3.
# Trailing run must be the 3 post-demotion faded rows, not 5 and not 1.
put("u1", "kc.a", "faded", False, T.format(1), p=0.6)
put("u1", "kc.a", "faded", True, T.format(2), p=0.6)
put("u1", "kc.a", "solo", False, T.format(3), p=0.6)
put("u1", "kc.a", "faded", False, T.format(4), p=0.6)
put("u1", "kc.a", "faded", False, T.format(5), p=0.6)
put("u1", "kc.a", "faded", True, T.format(6), p=0.6)

# u2 on kc.a: a 5-serve low-p run in the MIDDLE (maximal, not trailing),
# spelled with the legacy stage name; ends above both flag floors' reach.
for i, (p, c) in enumerate([(0.2, False), (0.3, False), (0.1, False), (0.2, True), (0.3, False)]):
    put("u2", "kc.a", "partial", c, T.format(10 + i), p=p)
put("u2", "kc.a", "solo", True, T.format(20), p=0.9)

# u3 on kc.b: a 10-attempt faded stall at the exact flag floor.
for i in range(10):
    put("u3", "kc.b", "faded", i % 5 == 0, T.format(30 + i), p=0.4)

# A lesson view and an unrecognised stage must not count as evidence.
attempt_log.record_lesson_view("u3", "kc.b", base_dir=BASE)
put("u3", "kc.b", "warmup", True, T.format(50), p=0.9)

out = kc_stats.kc_stats(base_dir=BASE)
a, b = out["kcs"]["kc.a"], out["kcs"]["kc.b"]

print("aggregation")
check("three learners seen", out["learners"] == 3, out["learners"])
check("kc.a spans two learners", a["learners"] == 2, a["learners"])
check("kc.a attempts", a["attempts"] == 12, a["attempts"])
check("legacy 'partial' folded into faded",
      a["stages"].get("faded", {}).get("n") == 10, a["stages"])
check("kc.b ignores the lesson view and the bad stage",
      b["attempts"] == 10, b["attempts"])
check("kc.b accuracy", abs(b["accuracy"] - 0.2) < 1e-9, b["accuracy"])
check("brier present", a["brier"] is not None and 0 <= a["brier"] <= 1, a["brier"])

print("stall")
check("demotion restarts the trailing run",
      a["worst_stall"]["length"] == 3 and a["worst_stall"]["stage"] == "faded",
      a["worst_stall"])
check("stall accuracy is the run's own",
      abs(a["worst_stall"]["accuracy"] - 1 / 3) < 1e-9, a["worst_stall"])
check("kc.b stall at the floor", b["worst_stall"]["length"] == 10, b["worst_stall"])

print("low-p runs")
check("maximal run found mid-history, not trailing",
      a["worst_low_p_run"]["length"] == 5, a["worst_low_p_run"])
check("run mean p", abs(a["worst_low_p_run"]["mean_predicted_p"] - 0.22) < 1e-9,
      a["worst_low_p_run"])

print("flags")
kinds = {(f["kind"], f["kc"]) for f in out["flags"]}
check("kc.b stall flagged at floor", ("rung_stall", "kc.b") in kinds, kinds)
check("kc.a stall of 3 NOT flagged", ("rung_stall", "kc.a") not in kinds, kinds)
check("kc.a low-p run of 5 flagged at floor",
      ("served_while_predicting_failure", "kc.a") in kinds, kinds)
check("no user ids anywhere in payload",
      "u1" not in str(out) and "u3" not in str(out))

print(f"\n{len(fails)} failure(s)")
sys.exit(1 if fails else 0)
