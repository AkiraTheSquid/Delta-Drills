#!/usr/bin/env python3
"""course_pace_report.py — LeetCode vs ARENA, concept by concept, from the
ability model alone (Seth, 2026-10-02: do the math "on a per node basis for
leetcode, then compare it to the ARENA, keeping in mind that the time per
problem and difficulty is different" — check the gut AFTER).

For a fresh learner at each self-reported level, per concept: the question
pool's difficulty range, the concept's worth (the ability it demands), the
model's expected problems to READY after the lesson (app/learning_pace.py),
the minutes they take (MINUTES_BY_TIER — assumptions, see that module), and
so XP per problem and XP per hour. Then course totals side by side.

`--user <id>`: also that learner's remaining problems and minutes now, and
what a finish date asks per day.

Run: .venv/bin/python scripts/course_pace_report.py [--detail] [--user ID --by 2026-11-01]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from datetime import date, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import ability_model as A, bkt_mastery, course_registry, kc_graph  # noqa: E402
from app import learning_pace, learning_xp  # noqa: E402

LEVELS = (("beginner", "beginner"), ("no level", None), ("strong", "strong"))


def fresh(level):
    p = float(bkt_mastery.params_for_level(level).p_init)
    return {"post": A.prior(p), "R": 1.0, "lesson_seen": False}


def rows(course: str, level):
    out = []
    for kc in kc_graph._registry():
        if course_registry.course_of(kc) != course:
            continue
        n, mins, reached = learning_pace.course(kc, fresh(level))
        pool = A.pool(kc)
        out.append({"kc": kc, "lo": pool[0], "hi": pool[-1], "n_pool": len(pool), "reached": reached,
                    "worth": learning_xp.worth(kc), "problems": n, "minutes": mins})
    return out


def course_table(detail: bool):
    for course in ("leetcode", "arena"):
        print(f"\n=== {course} ===")
        for name, level in LEVELS:
            every = rows(course, level)
            rs = [r for r in every if r["reached"]]
            if not rs:
                print(f"{name:9s} {len(every):3d} concepts, every one out of reach")
                continue
            n = sum(r["problems"] for r in rs)
            m = sum(r["minutes"] for r in rs)
            xp = sum(r["worth"] for r in rs)
            print(f"{name:9s} {len(rs):3d} concepts  worth {xp:5d}  problems {n:6.0f} "
                  f"(median {statistics.median(r['problems'] for r in rs):4.1f}/concept)  "
                  f"hours {m / 60:6.1f}  XP/problem {xp / max(n, 1):5.1f}  XP/hour {xp / max(m / 60, 1e-9):6.1f}  "
                  f"min/problem {m / max(n, 1):4.1f}  out of reach {len(every) - len(rs)}")
        if detail:
            print(f"{'concept':44s} {'pool':>9s} {'n':>3s} {'worth':>5s} {'probs':>5s} {'min':>5s} {'XP/prob':>7s}")
            for r in sorted(rows(course, None), key=lambda r: -r["worth"]):
                print(f"{r['kc'][:44]:44s} {r['lo']:4.0f}-{r['hi']:<4.0f} {r['n_pool']:3d} {r['worth']:5d} "
                      + (f"{r['problems']:5.1f} {r['minutes']:5.0f} {r['worth'] / max(r['problems'], 1):7.1f}"
                         if r["reached"] else "    -     -       -  OUT OF REACH"))


def one_user(uid: str, by: date):
    from app import adaptive
    st = adaptive._load_user_state(uid)
    s = learning_xp.summary(st, timezone.utc)
    days = max(1, (by - date.today()).days + 1)  # today and the finish day both count, as in the app
    c = s["course"]
    print(f"\n=== learner {uid[:8]} — {c['name']}, {c['concepts']} concepts, {c['ready_concepts']} at READY ===")
    print(f"XP left {s['remaining']}  problems left {s['problems_remaining']}  "
          f"minutes left {s['minutes_remaining']} (≈ {s['minutes_remaining'] / 60:.1f} h)  "
          f"out of reach {s['concepts_out_of_reach']}")
    print(f"by {by} ({days} days): {s['remaining'] / days:.1f} XP/day, "
          f"{s['problems_remaining'] / days:.1f} problems/day, {s['minutes_remaining'] / days:.0f} min/day")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--user")
    ap.add_argument("--by", default=None)
    a = ap.parse_args()
    course_table(a.detail)
    if a.user:
        one_user(a.user, date.fromisoformat(a.by) if a.by else date.today())


if __name__ == "__main__":
    main()
