"""Invariants for app/learning_xp.py, run by app/watch.py (kept apart so
that file stops growing)."""


def check_the_finish_pace_is_net_and_counts_today():
    """The projected finish divides what is left by NET learning, today
    included (Seth, 2026-09-27: "Finish … isn't actually getting updated").
    A mean of daily XP counted only gains, skipped today, and so sat still
    all day and stayed optimistic through idle days."""
    from datetime import date, datetime, timedelta, timezone
    from app import learning_xp as lx
    utc = timezone.utc
    days = [date(2026, 9, 1) + timedelta(days=i) for i in range(20)]
    # 100 at the first open, +50 a day for 10 days, then 5 idle days losing
    # 10 each, then today (day 19) so far.
    closing = []
    k = 100.0
    for i in range(20):
        k += 50 if i < 10 else (-10 if i < 15 else 0)
        closing.append(k)
    r = {"days": days, "knowledge": closing, "open_knowledge": 100.0}
    noon = datetime(2026, 9, 20, 12, tzinfo=utc)
    start = closing[20 - 1 - lx.PACE_DAYS - 1]  # close of the day before the window
    want = (closing[-1] - start) / (lx.PACE_DAYS + 0.5)
    got = lx._pace(r, utc, noon)
    assert abs(got - want) < 1e-9, f"pace {got} != net {want} over the window + half of today"
    later = dict(r, knowledge=closing[:-1] + [closing[-1] + 40])
    assert lx._pace(later, utc, noon) > got, "today's learning did not move the pace"
    # …and so the finish date: remaining shrinks AND pace grows.
    today, total = days[-1], 2000.0
    before = lx._finish(today, total - closing[-1], got)
    after = lx._finish(today, total - closing[-1] - 40, lx._pace(later, utc, noon))
    assert after < before, f"an answer today did not bring the finish forward ({before} -> {after})"
    assert lx._finish(today, 0, got) is None and lx._finish(today, 100, 0) is None
    losing = dict(r, knowledge=[100.0 - i for i in range(20)])
    assert lx._pace(losing, utc, noon) == 0.0, "a net loss must read as no pace, not a negative one"
    one = {"days": [days[0]], "knowledge": [160.0], "open_knowledge": 100.0}
    assert lx._pace(one, utc, datetime(2026, 9, 1, 1, tzinfo=utc)) == 60.0, \
        "an hour-old history must be priced over a whole day"


def check_a_concept_is_priced_by_what_builds_on_it():
    """Concepts are worth different XP, foundations most (Seth, 2026-09-29:
    "not all concepts are equally important"), and one solved problem reads
    on the order of 5-15 XP, not a concept's whole 80. No levels."""
    from app import concept_choice as cc
    from app import learning_xp as lx
    worths = [lx.worth_of(n) for n in range(0, 200)]
    assert worths[0] == lx.WORTH_LEAF and worths[-1] == 40, worths[:3]
    assert all(a <= b for a, b in zip(worths, worths[1:])), "more descendants must never be worth less"
    assert all(w % lx.WORTH_STEP == 0 for w in worths)
    assert lx.worth("no.such-concept") == lx.WORTH_LEAF, "an unknown concept is priced as a leaf"
    # Hardness (a LeetCode learner, 2026-10-01: recursion is much harder than
    # linked lists, same XP?): harder pays more, inside the bounds, and the
    # same concept count of descendants can no longer tie them.
    assert lx.worth_of(0, lx.HARD_MIN) == lx.WORTH_MIN and lx.worth_of(199, lx.HARD_MAX) == lx.WORTH_MAX
    assert all(lx.worth_of(n, 0.8) <= lx.worth_of(n, 1.3) for n in range(0, 200))
    assert lx.hardness("no.such-concept") == 1.0
    assert lx.rates("no.such-concept") == (lx.T_LESSON, lx.T_ANSWER, lx.T_AIDED)
    # A novice's first solved problem, lesson included, on a leaf and on a
    # foundation: a handful of XP, the foundation more.
    novice = {"p": 0.02, "R": 1.0, "lesson_seen": False}
    kc = "no.such-concept"
    leaf = lx.solve_xp(kc, novice)
    assert 0 < leaf <= lx.WORTH_LEAF * 0.75, f"a leaf's first problem paid {leaf}"
    worn = {"p": 0.9, "R": 0.5, "lesson_seen": True}
    assert lx.solve_xp(kc, worn) > 0, "relearning after forgetting must still pay"
    done = {"p": 0.99, "R": 1.0, "lesson_seen": True}
    assert lx.solve_xp(kc, done) == 0.0, "past READY a problem pays nothing"
    assert cc.per_problem(0.0) == 0 and cc.per_problem(0.2) == 5 and cc.per_problem(12.4) == 10
    assert cc.per_problem(12.6) == 15, "per-problem XP shows to the nearest 5"


def check_the_planner_counts_problems_by_the_same_model():
    """XP a day is not problems a day (a LeetCode learner, 2026-10-01, read a
    one-month target of 20 XP/day beside "+15 XP / problem" as two problems a
    day). app/learning_pace counts the problems with the model that pays the
    XP: none past READY, more on a harder concept, deterministic."""
    from app import learning_pace as lp
    from app import learning_xp as lx
    novice = {"p": 0.02, "R": 1.0, "lesson_seen": False}
    assert lp.problems_to_ready("no.such-concept", {"p": 0.99, "R": 1.0}) == 0.0
    orig = lx.hardness
    try:  # uncached, so the stand-in hardness is what is counted
        lx.hardness = lambda kc: {"x.easy": lx.HARD_MIN, "x.hard": lx.HARD_MAX}.get(kc, 1.0)
        easy = lp._count.__wrapped__("x.easy", 0.02, 1.0, False)
        hard = lp._count.__wrapped__("x.hard", 0.02, 1.0, False)
    finally:
        lx.hardness = orig
    assert easy < hard, f"a harder concept must take more problems ({easy} vs {hard})"
    n = lp.problems_to_ready("no.such-concept", novice)
    assert n >= 3 and n == lp.problems_to_ready("no.such-concept", novice), n
    assert lp.problems_remaining({"no.such-concept": novice, "x.done": {"p": 0.99, "R": 1.0}}) == n


if __name__ == "__main__":
    check_the_finish_pace_is_net_and_counts_today()
    check_a_concept_is_priced_by_what_builds_on_it()
    check_the_planner_counts_problems_by_the_same_model()
