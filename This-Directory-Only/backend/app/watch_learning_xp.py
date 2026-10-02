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


def check_a_concept_is_priced_by_the_ability_it_demands():
    """A concept's worth is computed, not chosen (Seth, 2026-10-02: "it needs
    to be computed rather than hardcoded"): the ability at which the model
    expects 80% of its pool solved. Harder pool, more worth; a solved problem
    pays by the learner's state, most mid-way, nothing past READY."""
    from app import ability_model as am
    from app import concept_choice as cc
    from app import learning_xp as lx
    assert am._ready_theta((40.0,)) < am._ready_theta((40.0, 90.0)) < am._ready_theta((90.0,))
    assert lx.worth("no.such-concept") == round(am._ready_theta((am.DIFFICULTY_MID,)))
    novice = {"post": am.prior(0.02), "R": 1.0, "lesson_seen": False}
    first = lx.solve_xp("no.such-concept", novice)
    assert 0 < first < lx.worth("no.such-concept") / 2, f"a novice's first problem paid {first}"
    worn = {"post": am._prior(am._ready_theta((am.DIFFICULTY_MID,)) + 15), "R": 0.5, "lesson_seen": True}
    assert lx.solve_xp("no.such-concept", worn) > 0, "relearning after forgetting must still pay"
    done = {"post": am._prior(130.0), "R": 1.0, "lesson_seen": True}
    assert lx.solve_xp("no.such-concept", done) == 0.0, "past READY a problem pays nothing"
    # Learning is fastest at the learner's level and ~0 far from it.
    z = am.zpd(60.0)
    i60, i10, i120 = (int(abs(am.GRID - x).argmin()) for x in (60.0, 10.0, 120.0))
    assert z[i60] > 0.99 and z[i10] < 0.03 and z[i120] < 0.03
    assert cc.per_problem(0.0) == 0 and cc.per_problem(0.2) == 5 and cc.per_problem(12.4) == 10
    assert cc.per_problem(12.6) == 15, "per-problem XP shows to the nearest 5"


def check_the_planner_counts_problems_by_the_same_model():
    """XP a day is not problems a day (a LeetCode learner, 2026-10-01, read a
    one-month target of 20 XP/day beside "+15 XP / problem" as two problems a
    day). app/learning_pace counts each concept's problems with the model that
    pays the XP: none past READY, one for a faded concept, more on a harder
    pool, more for a beginner, deterministic, minutes by course."""
    from app import ability_model as am
    from app import learning_pace as lp
    novice = {"post": am.prior(0.02), "R": 1.0, "lesson_seen": False}
    strong = {"post": am.prior(0.45), "R": 1.0, "lesson_seen": False}
    pools = {"x.mid": tuple(float(d) for d in range(15, 101, 5)),
             "x.easy": tuple(float(d) for d in range(15, 61, 5)),
             "x.lone": (50.0,)}
    orig = am.pool
    try:
        am.pool = lambda kc: pools.get(kc, (50.0,))
        ready = am.ready_theta("x.mid")
        assert lp.problems_to_ready("x.mid", {"post": am._prior(135.0), "R": 1.0}) == 0.0
        assert lp.problems_to_ready("x.mid", {"post": am._prior(ready + 20), "R": 0.3}) == 1.0
        n = lp.problems_to_ready("x.mid", novice)
        assert 3 <= n < am.CAP and n == lp.problems_to_ready("x.mid", novice), n
        assert lp.problems_to_ready("x.mid", strong) < n
        assert lp.problems_to_ready("x.easy", novice) < n, "a harder pool must take more problems"
        # One lone problem far above a novice teaches nothing: out of reach,
        # counted apart rather than as the fuse's CAP problems.
        assert lp.course("x.lone", novice)[2] is False
        left = lp.remaining({"x.mid": novice, "x.lone": novice, "x.done": {"post": am._prior(135.0), "R": 1.0}})
        assert left[0] == n and left[2] == ["x.lone"], left
    finally:
        am.pool = orig
    assert lp.minutes("leetcode.recursion", 90) > lp.minutes("leetcode.recursion", 20) > lp.minutes("torch.x", 20)


if __name__ == "__main__":
    check_the_finish_pace_is_net_and_counts_today()
    check_a_concept_is_priced_by_the_ability_it_demands()
    check_the_planner_counts_problems_by_the_same_model()
