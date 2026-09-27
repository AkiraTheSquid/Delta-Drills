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


if __name__ == "__main__":
    check_the_finish_pace_is_net_and_counts_today()
