"""learning_xp.py — XP as MEASURED LEARNING, not effort.

WHY THIS EXISTS
---------------
Until 2026-09-26 XP was a localStorage counter: 25 for a correct answer, 10
for a miss, 1 per 15 s of typing. A 25-minute solve and a 3-minute guess paid
the same, and nothing about it said whether the learner was any closer to the
course. Seth, 2026-09-26: XP should be "measured by your actual learning
progress", the same yardstick for every learner — "10% increase in ability
for a concept is the same exp for every learner" — so a slower learner needs
MORE problems for the same XP, never a different price for the course.

THE UNIT
--------
1 XP = one percentage point of knowledge on one concept, counted up to the
READY line (80%). The course is every concept in the ARENA graph through 0.2,
prerequisites included (`study_group_progress.sections()`), so the whole
course costs `len(scope) × 80` XP for everyone. A level is one concept's worth:
LEVEL_XP = 80.

KNOWLEDGE OF ONE CONCEPT
------------------------
K_c(t) = P(learned_c at t | ALL evidence) × R_c(t)

  * P(learned) is a two-state HMM, unlearned → learned, with guess/slip
    emissions from `kc_explore.likelihood` (MC for math, code otherwise). A
    learned concept that is not retrieved answers like an unlearned one, so
    the emission of the learned state is (1 − slip)·R + guess·(1 − R): a miss
    after a break is read as forgetting, not as never having learned it.
    This is the per-concept model of the 2026-09-24 planner design.
  * R is the FSRS-6 + FIRe retrievability from `memory_model`, replayed over
    the same answers. Time lowers R only; P(learned) moves only on evidence.

WHY IT IS SMOOTHED — THE MODEL LEARNING vs THE LEARNER LEARNING
-----------------------------------------------------------------
Seth: where the model is uncertain it must not read a jump in its own
estimate as learning. Explore answers mostly tell the MODEL where the learner
already was. So P(learned) at a day boundary is the forward-backward SMOOTHED
posterior, using every answer up to now, and each answer carries a learning
rate by what surrounded it:

  * explore probe (no lesson, no example — a ladder row flagged `probe`, or a
    placement probe): T_PROBE, near zero. A correct probe raises the smoothed
    belief at the START of the day as much as at the end — it is starting
    credit, not XP.
  * exploit answer (lessons and worked examples available): the first one on
    a concept is preceded by its lesson (T_LESSON), and every answer is a
    practice step (T_ANSWER). Missed, read the lesson, then solved 5 of 7 →
    the change of state lands inside the day and it is XP.
  * an answer made behind a worked example is restudy with the answer in
    view: a learning step, but no evidence.

Later answers revise earlier days, which is intended: as the sample size on a
concept grows, the model gets surer WHEN the learner learned it. Today is
reported as provisional for the same reason.

DAILY XP
--------
XP_day = Σ_c 100 · max(0, min(K_c(close), READY) − min(K_c(open), READY))

Floored per concept per day: forgetting never removes XP already earned, and
relearning what a break took away (R back up) earns it again — Seth: FSRS
still decays, but progress toward the goal is still rewarded. `remaining` is
read off today's K, so forgetting raises it honestly. What CAN move a past
day's XP is new evidence: the smoothing re-reads the whole chain, so a run of
later misses can lower the belief that a concept was learned on Tuesday, and
Tuesday's XP (and the level) with it. That is the model correcting itself,
not the learner losing anything, and it is why today is provisional.

STATELESS, like `memory_model`: replayed from the ladder, the attempt log and
the placement probes on every read. Nothing here writes practice state except
`set_target`.
"""
from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from app import attempt_log, bkt_mastery, kc_explore, kc_renames, memory_model

READY = 0.80
XP_PER_CONCEPT = int(round(READY * 100))
LEVEL_XP = XP_PER_CONCEPT

# Learning rates, P(unlearned → learned), per opportunity. v0 values, not a
# fit — fit from pooled logs with the rest of the planner model. Anchored on
# the ladder's own pacing (`kc_ladder_math.PROMOTE_LO`: four consecutive
# correct answers promote off Solo): from a novice prior, the lesson plus ONE
# correct Solo answer is ~40% of the concept, three or four in a row reach the
# READY line, and five misses with one correct stay low. An explore probe is
# nearly pure measurement.
T_PROBE = 0.03
T_LESSON = 0.15
T_ANSWER = 0.10
T_AIDED = 0.10

# P(correct | NOT learned). An explore probe has no lesson and no example in
# front of it, so it keeps `kc_explore.likelihood` (MC 0.25, code 0.10). An
# exploit answer is made with the lesson and its worked examples in reach —
# a pattern can be copied without the concept — so its guess is floored at
# GUESS_EXPLOIT, and higher on a scaffolded rung (`faded` = Faded,
# `worked` = the lesson page; `partial` is the unscaffolded Solo rung and
# `solo` Integrated, the ids are historical).
GUESS_EXPLOIT = 0.30
GUESS_BY_STAGE = {"worked": 0.50, "faded": 0.45}

# The projection's pace: net learning over the last this-many days plus
# today so far.
PACE_DAYS = 14

TARGET_MODES = ("date", "daily")


# --- scope -------------------------------------------------------------------

def scope_kcs() -> List[str]:
    """Every concept through ARENA 0.2, prerequisites included, in section
    order. Deliberately the same list for every learner: the price of the
    course may not depend on who is paying it."""
    from app import study_group_progress  # heavy import; only on use
    out: List[str] = []
    for area in study_group_progress.sections():
        out.extend(area["kcs"])
    return list(dict.fromkeys(out))


def _prior(user_state) -> float:
    level = getattr(user_state, "self_reported_level", None)
    return float(bkt_mastery.params_for_level(level).p_init)


# --- the evidence stream -----------------------------------------------------

# Where an answer's metadata comes from, best first: the log's own probe flag,
# the ladder row (inside its 20-row window), what the log recorded about the
# lesson. See `_answer_meta`.
_FLAGGED, _LADDER, _INFERRED = 0, 1, 2


def _answer_meta(user_state) -> Dict[Tuple[str, Optional[int]], List[tuple]]:
    """(kc, question) → [(t, source, probe?, ladder stage)] for every answer.

    The attempt log is the durable record: since 2026-09-26 each row carries
    the probe flag (`AttemptRow.probe`). For an older row the ladder row
    supplies it while it is still inside its 20-row window. Past that, the
    log's `days_since_read` decides: an answer made before the concept's
    lesson was ever read measured what the learner came with, which is what a
    probe is. A row older than that field (2026-09-19) predates explore
    probing (2026-09-22) and is exploit. Joined per question and by nearest
    time, never by a rounded second: two quick answers must not merge."""
    meta: Dict[Tuple[str, Optional[int]], List[tuple]] = {}
    uid = getattr(user_state, "user_id", None)
    if isinstance(uid, str) and uid:
        for r in attempt_log.iter_rows(uid):
            if r.kind != attempt_log.KIND_ATTEMPT or not r.kc:
                continue
            t = memory_model._to_days(r.ts)
            if t is None:
                continue
            if r.probe is not None:
                src, probe = _FLAGGED, bool(r.probe)
            else:
                sources = r.feature_sources if isinstance(r.feature_sources, dict) else {}
                probe = "days_since_read" in sources and sources["days_since_read"] is None
                src = _INFERRED
            meta.setdefault((kc_renames.canon(r.kc), r.question_id), []).append((t, src, probe, r.stage))
    for kc, row in (getattr(user_state, "kc_ladder", None) or {}).items():
        if not isinstance(row, dict):
            continue
        for att in row.get("attempts") or []:
            if not isinstance(att, dict):
                continue
            t = memory_model._to_days(att.get("ts") or att.get("timestamp"))
            if t is None:
                continue
            probe = bool(att.get("probe")) and not att.get("example")
            meta.setdefault((kc, att.get("question_id")), []).append((t, _LADDER, probe, att.get("stage")))
    return meta


def _meta_for(meta, kc: str, qid: Optional[int], t: float) -> Tuple[bool, Optional[str]]:
    """(probe?, stage) of the answer at `t`: the best source within the
    ladder↔log matching window (`memory_model._SAME_ANSWER_DAYS`), nearest
    first. None found = exploit."""
    near = [m for m in meta.get((kc, qid), ()) if abs(m[0] - t) <= memory_model._SAME_ANSWER_DAYS]
    if not near:
        return False, None
    _t, _src, probe, stage = min(near, key=lambda m: (m[1], abs(m[0] - t)))
    return probe, stage


def _likelihood(kc: str, stage: Optional[str], probe: bool) -> Tuple[float, float]:
    """(guess, slip) for one answer — see GUESS_EXPLOIT."""
    g, s = kc_explore.likelihood(kc)
    if probe:
        return g, s
    return max(g, GUESS_EXPLOIT, GUESS_BY_STAGE.get(stage or "", 0.0)), s


def _placement_probes(user_state) -> List[Tuple[float, str, bool]]:
    diag = getattr(user_state, "diagnostic", None)
    probes = diag.get("probes") if isinstance(diag, dict) else None
    out = []
    for p in probes if isinstance(probes, list) else []:
        if not isinstance(p, dict) or not p.get("kc"):
            continue
        t = memory_model._to_days(p.get("ts"))
        if t is None:
            continue
        # `dont_know` is a response, and it is not a correct one.
        out.append((t, kc_renames.canon(p["kc"]), p.get("result") == "correct"))
    return out


# --- the HMM -----------------------------------------------------------------

def _trans(T: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return ((1.0 - T, T), (0.0, 1.0))


def _smooth(prior: float, obs: List[tuple]) -> List[float]:
    """Smoothed P(learned) at every slot 0..n, given ALL of `obs`.

    Slot j is the state after observation j. Each observation is
    (pre_T, correct|None, R, guess, slip, post_T): a learning step before the
    answer (the lesson), the answer emitted from the state it found, and a
    learning step after it (practice). `correct is None` = no evidence."""
    n = len(obs)
    fwd = [(1.0 - prior, prior)]
    emits = []
    for pre_T, correct, R, g, s, post_T in obs:
        u, l = fwd[-1]
        a = _trans(pre_T)
        u, l = u * a[0][0], u * a[0][1] + l
        if correct is None:
            e = (1.0, 1.0)
        else:
            p_l = (1.0 - s) * R + g * (1.0 - R)
            e = (g, p_l) if correct else (1.0 - g, 1.0 - p_l)
        emits.append(e)
        u, l = u * e[0], l * e[1]
        z = u + l or 1.0
        u, l = u / z, l / z
        b = _trans(post_T)
        fwd.append((u * b[0][0], u * b[0][1] + l))
    bwd = [(1.0, 1.0)] * (n + 1)
    for j in range(n, 0, -1):
        pre_T, _c, _R, _g, _s, post_T = obs[j - 1]
        bu, bl = bwd[j]
        b = _trans(post_T)
        # through the post-answer step, then the emission, then the pre step
        xu, xl = b[0][0] * bu + b[0][1] * bl, bl
        e = emits[j - 1]
        xu, xl = xu * e[0], xl * e[1]
        a = _trans(pre_T)
        yu, yl = a[0][0] * xu + a[0][1] * xl, xl
        z = yu + yl or 1.0
        bwd[j - 1] = (yu / z, yl / z)
    out = []
    for (fu, fl), (bu, bl) in zip(fwd, bwd):
        z = fu * bu + fl * bl
        out.append(fl * bl / z if z > 0 else fl)
    return out


# --- replay ------------------------------------------------------------------

def _midnight(day: date, zone) -> datetime:
    return datetime.combine(day, time.min, zone)


def replay(user_state, zone, now: Optional[datetime] = None) -> dict:
    """Per-day XP and knowledge from the learner's whole history.

    Returns {"days": [date...], "xp": [per-day XP], "knowledge": [course
    knowledge XP at each day's close], "open_knowledge": knowledge at the first
    day's open, "answers": [answers per day]}."""
    now = now or datetime.now(timezone.utc)
    scope = scope_kcs()
    in_scope = set(scope)
    prior = _prior(user_state)
    cfg = memory_model.DEFAULT_CONFIG
    meta = _answer_meta(user_state)

    stream: List[tuple] = []  # (t_days, kind, payload)
    for ev in memory_model._events(user_state):
        stream.append((ev.t, 0, ev))
    for t, kc, ok in _placement_probes(user_state):
        stream.append((t, 1, (kc, ok)))
    stream.sort(key=lambda x: (x[0], x[1]))

    today = now.astimezone(zone).date()
    first = today
    if stream:
        first_dt = datetime.fromtimestamp(stream[0][0] * 86400.0, timezone.utc)
        first = min(today, first_dt.astimezone(zone).date())
    n_days = (today - first).days + 1
    # Boundary i is the open of day i; the last one is `now` (today's close
    # so far).
    bounds = [_midnight(first + timedelta(days=i), zone) for i in range(n_days)] + [now]
    bound_t = [b.timestamp() / 86400.0 for b in bounds]

    mems: Dict[str, memory_model.Memory] = {}
    obs: Dict[str, List[tuple]] = {kc: [] for kc in scope}
    seen_exploit: set = set()
    snap_R: List[Dict[str, float]] = []
    snap_j: List[Dict[str, int]] = []
    answers = [0] * n_days

    def snapshot(t: float) -> None:
        snap_R.append({kc: memory_model.retrievability(m, t, cfg)
                       for kc, m in mems.items() if kc in in_scope})
        snap_j.append({kc: len(o) for kc, o in obs.items() if o})

    def observe(kc: str, t: float, correct: Optional[bool], probe: bool,
                stage: Optional[str] = None) -> None:
        g, s = _likelihood(kc, stage, probe)
        R = memory_model.retrievability(mems[kc], t, cfg) if kc in mems else 1.0
        if probe:
            obs[kc].append((0.0, correct, R, g, s, T_PROBE))
            return
        pre = 0.0
        if kc not in seen_exploit:
            seen_exploit.add(kc)
            pre = T_LESSON
        obs[kc].append((pre, correct, R, g, s, T_AIDED if correct is None else T_ANSWER))

    bi = 0
    for t, kind, payload in stream:
        while bi < len(bound_t) and bound_t[bi] <= t:
            snapshot(bound_t[bi])
            bi += 1
        # bound_t[bi - 1] <= t < bound_t[bi]: the answer falls in day bi - 1.
        if 1 <= bi <= n_days:
            answers[bi - 1] += 1
        if kind == 1:
            kc, ok = payload
            if kc in in_scope:
                observe(kc, t, ok, probe=True)
                # The placement answer is a retrieval like any other: without
                # it a concept known only from placement kept R = 1 forever
                # and its starting credit never faded (codex, 2026-09-26).
                # Local to this replay; the app's own memory is unchanged.
                grade = memory_model.GOOD if ok else memory_model.AGAIN
                memory_model._apply(mems, t, {kc: grade}, cfg, None)
            continue
        ev = payload
        for kc, grade in ev.grades.items():
            if kc not in in_scope:
                continue
            correct = None if grade == memory_model.AIDED else grade != memory_model.AGAIN
            probe, stage = _meta_for(meta, kc, ev.qid, t)
            observe(kc, t, correct, probe, stage)
        memory_model._apply(mems, t, ev.grades, cfg, ev.p_skill)
    while bi < len(bound_t):
        snapshot(bound_t[bi])
        bi += 1

    smoothed = {kc: _smooth(prior, o) for kc, o in obs.items() if o}

    def knowledge(b: int, kc: str) -> float:
        sm = smoothed.get(kc)
        p = sm[snap_j[b].get(kc, 0)] if sm else prior
        return p * snap_R[b].get(kc, 1.0)

    capped = [[min(knowledge(b, kc), READY) for kc in scope] for b in range(len(bounds))]
    xp, closing = [], []
    for i in range(n_days):
        gain = sum(max(0.0, c1 - c0) for c0, c1 in zip(capped[i], capped[i + 1]))
        xp.append(100.0 * gain)
        closing.append(100.0 * sum(capped[i + 1]))
    per_kc_now = {kc: knowledge(len(bounds) - 1, kc) for kc in scope}
    return {
        "scope": scope,
        "days": [first + timedelta(days=i) for i in range(n_days)],
        "xp": xp,
        "knowledge": closing,
        "open_knowledge": 100.0 * sum(capped[0]),
        "today_open_knowledge": 100.0 * sum(capped[n_days - 1]),
        "answers": answers,
        "per_kc_now": per_kc_now,
    }


# --- the readout -------------------------------------------------------------

def _pace(r: dict, zone, now: datetime) -> float:
    """Course knowledge gained per day, net of forgetting, from the open of
    the day PACE_DAYS before today up to `now`. Net, because `remaining`
    shrinks by net learning: a mean of the daily XP (gross gains only) kept
    the projection optimistic through every idle day. Today counts, so the
    finish date moves with each answer instead of once at midnight."""
    days, closing = r["days"], r["knowledge"]
    k = max(0, len(days) - 1 - PACE_DAYS)
    start = r["open_knowledge"] if k == 0 else closing[k - 1]
    elapsed = (now - _midnight(days[k], zone)).total_seconds() / 86400.0
    # At least a day: an hour-old history would otherwise project from one
    # good hour.
    return max(0.0, closing[-1] - start) / max(1.0, elapsed)


def _finish(today: date, remaining: float, pace: float) -> Optional[str]:
    """The day `remaining` reaches zero at `pace`; None with no pace or
    nothing left."""
    if pace <= 0 or remaining <= 0:
        return None
    return (today + timedelta(days=int(math.ceil(remaining / pace)))).isoformat()


def daily_target(target: Optional[dict], remaining_open: float, today: date) -> Optional[int]:
    """XP needed today. `date` mode spreads what was left at today's open over
    the days up to and including the target date, re-read every day, so a
    missed day raises tomorrow's number and forgetting is priced in."""
    if not isinstance(target, dict):
        return None
    if target.get("mode") == "daily":
        daily = target.get("daily")
        return int(daily) if isinstance(daily, (int, float)) and daily > 0 else None
    if target.get("mode") == "date":
        try:
            end = date.fromisoformat(str(target.get("date")))
        except ValueError:
            return None
        if end < today:
            return None  # the date has passed; the learner sets a new one
        days_left = (end - today).days + 1
        return int(math.ceil(max(0.0, remaining_open) / days_left))
    return None


def summary(user_state, zone, now: Optional[datetime] = None) -> dict:
    now = now or datetime.now(timezone.utc)
    r = replay(user_state, zone, now)
    total = len(r["scope"]) * XP_PER_CONCEPT
    earned = sum(r["xp"])
    level = 1 + int(earned // LEVEL_XP)
    know_now = r["knowledge"][-1]
    remaining = max(0.0, total - know_now)
    today = r["days"][-1]
    target = getattr(user_state, "xp_target", None) or None
    remaining_open = max(0.0, total - r["today_open_knowledge"])
    need_today = daily_target(target, remaining_open, today)
    pace = _pace(r, zone, now)
    finish = _finish(today, remaining, pace)
    ready = sum(1 for v in r["per_kc_now"].values() if v >= READY - 1e-9)
    return {
        "course": {"concepts": len(r["scope"]), "total_xp": total, "ready_at": XP_PER_CONCEPT,
                   "through": "0.2", "ready_concepts": ready},
        "earned": round(earned, 1),
        "level": level,
        "into": round(earned - (level - 1) * LEVEL_XP, 1),
        "need": LEVEL_XP,
        "knowledge": round(know_now, 1),
        "starting_credit": round(r["open_knowledge"], 1),
        "remaining": round(remaining, 1),
        # What a date target divides: the remainder at today's OPEN, so today's
        # own learning does not shrink today's number. The form's preview uses it.
        "remaining_open": round(remaining_open, 1),
        "today": {"date": today.isoformat(), "xp": round(r["xp"][-1], 1),
                  "target": need_today, "provisional": True},
        "target": target,
        "pace": round(pace, 1),
        "pace_days": PACE_DAYS,
        "projected_finish": finish,
        "days": [{"date": d.isoformat(), "xp": round(x, 1), "knowledge": round(k, 1), "answers": a}
                 for d, x, k, a in zip(r["days"], r["xp"], r["knowledge"], r["answers"])],
    }


def set_target(user_state, mode: str, day: Optional[str] = None, daily: Optional[int] = None,
               today: Optional[date] = None) -> Optional[dict]:
    """Store the learner's target. `mode` "none" clears it. Raises ValueError
    with a sentence the UI can show."""
    if mode == "none":
        user_state.xp_target = None
        return None
    if mode == "daily":
        if not isinstance(daily, int) or not 1 <= daily <= 2000:
            raise ValueError("Pick a daily XP between 1 and 2000.")
        user_state.xp_target = {"mode": "daily", "daily": daily}
        return user_state.xp_target
    if mode == "date":
        try:
            end = date.fromisoformat(str(day))
        except ValueError:
            raise ValueError("That date could not be read.") from None
        if end < (today or datetime.now(timezone.utc).date()):
            raise ValueError("Pick a date that has not passed.")
        user_state.xp_target = {"mode": "date", "date": end.isoformat()}
        return user_state.xp_target
    raise ValueError("Choose a finish date or a daily XP.")
