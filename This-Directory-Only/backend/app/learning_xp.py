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

THE UNIT — COMPUTED, NOT PRICED BY HAND
---------------------------------------
Knowledge of a concept is counted up to the READY line (80% of its problems),
and reaching it pays the concept's WORTH. History: every concept 80 XP
(09-26); 10 + 5·log2(1 + descendants), 10..40 (09-29: "not all concepts are
equally important"); that × hardness² (10-01: "recursion is much harder than
linked lists"). Seth, 2026-10-02, on all of them: "it needs to be computed
rather than hardcoded ... the numbers for how much xp you get fall out of the
model", and ARENA's "10xp or 15xp in a very arbitrary way". So since
2026-10-02:

  worth(c) = the ABILITY concept c demands, in difficulty points

— the ability θ at which the model expects 80% of c's question pool solved
(`ability_model.ready_theta`). Recursion's problems run harder than linked
lists', so it demands more (≈89 vs ≈70) and is worth more; nobody chose
either number. XP from a concept = worth × min(K, READY) / READY, the same
price for every learner. What ONE problem pays falls out of the learner's own
state (`solve_xp`): most at their level, almost nothing for a problem they
were always going to solve or could not, and nothing for a wrong answer
(see CREDIT). The course is every concept in the ARENA graph through 0.2,
prerequisites included (`study_group_progress.sections()`), or the studied
standalone course, and costs Σ worth for everyone. No levels. Importance in
the graph (coreness) is the PLANNER's business — what to do next — not the
price of what was learned.

KNOWLEDGE OF ONE CONCEPT
------------------------
K_c(t) = P(solving a novel problem on c at t | ALL evidence) × R_c(t)

  * The first factor is app/ability_model.py: an ability θ on the bank's
    difficulty scale, held as a distribution, answers emitted by difficulty
    (guess/slip from `kc_explore.likelihood`, MC for math, code otherwise),
    and learning that moves θ most on problems at the learner's level. A
    learner who has the skill but does not retrieve it answers like one who
    never had it: a miss after a break is read as forgetting.
  * R is the FSRS-6 + FIRe retrievability from `memory_model`, replayed over
    the same answers. Time lowers R only; θ moves only on evidence.

CREDIT — NO XP FOR A WRONG ANSWER
---------------------------------
Seth, 2026-10-02: "if the learner gets a problem wrong they don't get the xp
right?" Right. The XP ledger reads a concept's knowledge only as of its latest
CORRECT answer: the smoothed ability at that answer, times the recall left
from it. A miss moves nothing; the learning around it is paid when a correct
answer proves it. A miss does not take XP back either; later evidence can
(see below).

WHY IT IS SMOOTHED — THE MODEL LEARNING vs THE LEARNER LEARNING
-----------------------------------------------------------------
Seth: where the model is uncertain it must not read a jump in its own
estimate as learning. Explore answers mostly tell the MODEL where the learner
already was. So the ability at a day boundary is the forward-backward SMOOTHED
posterior, using every answer up to now, and each answer carries a learning
rate by what surrounded it:

  * explore probe (no lesson, no example — a ladder row flagged `probe`, or a
    placement probe): RATE_PROBE, near zero. A correct probe raises the
    smoothed belief at the START of the day as much as at the end — it is
    starting credit, not XP.
  * exploit answer (lessons and worked examples available): the first one on
    a concept is preceded by its lesson (RATE_LESSON), and every answer is a
    practice step (RATE_ANSWER) on that problem's difficulty when it is
    SOLVED — a miss teaches nothing (ability_model, LEARNING). Missed, read
    the lesson, then solved 5 of 7 → the change of state lands inside the day
    and it is XP.
  * an answer made behind a worked example is restudy with the answer in
    view: a learning step, but no evidence.

Later answers revise earlier days, which is intended: as the sample size on a
concept grows, the model gets surer WHEN the learner learned it. Today is
reported as provisional for the same reason.

DAILY XP
--------
XP_day = Σ_c worth(c)/READY · max(0, min(C_c(close), READY) − min(C_c(open), READY))

with C the CREDITED knowledge (see CREDIT).

Floored per concept per day: forgetting never removes XP already earned, and
relearning what a break took away (R back up) earns it again — Seth: FSRS
still decays, but progress toward the goal is still rewarded. `remaining` is
read off today's K, so forgetting raises it honestly. What CAN move a past
day's XP is new evidence: the smoothing re-reads the whole chain, so a run of
later misses can lower the belief that a concept was learned on Tuesday, and
Tuesday's XP with it. That is the model correcting itself,
not the learner losing anything, and it is why today is provisional.

STATELESS, like `memory_model`: replayed from the ladder, the attempt log and
the placement probes on every read. Nothing here writes practice state except
`set_target`.
"""
from __future__ import annotations

import math
from functools import lru_cache
from datetime import date, datetime, time, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from app import ability_model as A
from app import attempt_log, bkt_mastery, kc_explore, kc_renames, memory_model

READY = A.READY
READY_PCT = int(round(READY * 100))
# The course's name on the Learner Home (app/course_registry.py's label).
COURSE_NAME = "ARENA"

# P(correct | no skill). An explore probe has no lesson and no example in
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

def _standalone_study(user_state) -> List[str]:
    """The courses a learner studies when ARENA is NOT one of them, else [].

    🔴 Seth, 2026-10-01: a LeetCode-only learner's Learner Home still read
    "ARENA · through 0.2 · 125 concepts" — the course she was not studying,
    linear algebra and all, priced into her % complete. Her drills were
    already LeetCode (course_registry.studied); the home's yardstick was not.
    A learner who studies ARENA keeps the full list below, unchanged."""
    if user_state is None:
        return []
    from app import course_registry
    study = course_registry.studied(user_state)
    return [] if "arena" in study else study


def scope_kcs(user_state=None) -> List[str]:
    """Every concept through ARENA 0.2, prerequisites included, in section
    order. Deliberately the same list for every learner OF A COURSE: the
    price of the course may not depend on who is paying it. A learner who
    does not study ARENA (`_standalone_study`) gets only their courses'
    concepts, priced the same for everyone on that course."""
    from app import course_registry, study_group_progress  # heavy import; only on use
    out: List[str] = []
    for area in study_group_progress.sections():
        out.extend(area["kcs"])
    out = list(dict.fromkeys(out))
    study = _standalone_study(user_state)
    if study:
        # No fallback to the full list: that would price ARENA under the
        # standalone course's name, the very leak this scope closes.
        out = [kc for kc in out if course_registry.course_of(kc) in study]
    return out


def course_info(user_state=None) -> dict:
    """The Learner Home's course line: name and how far it runs. `through`
    is ARENA's chapter; a standalone course has none."""
    study = _standalone_study(user_state)
    if not study:
        return {"name": COURSE_NAME, "through": "0.2"}
    from app import course_registry
    labels = {c["id"]: c["label"] for c in course_registry.COURSES}
    return {"name": " + ".join(labels.get(c, c) for c in study), "through": None}


@lru_cache(maxsize=None)
def worth(kc: str) -> int:
    """XP a concept pays at READY: the ability it demands (see THE UNIT). A
    concept the bank has no questions for is one mid-difficulty problem."""
    return int(round(A.ready_theta(kc)))


def _prior(user_state):
    """The starting distribution over ability, from the self-reported level."""
    level = getattr(user_state, "self_reported_level", None)
    return A.prior(float(bkt_mastery.params_for_level(level).p_init))


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


# --- replay ------------------------------------------------------------------

def _midnight(day: date, zone) -> datetime:
    return datetime.combine(day, time.min, zone)


def replay(user_state, zone, now: Optional[datetime] = None, also: Tuple[str, ...] = ()) -> dict:
    """Per-day XP and knowledge from the learner's whole history.

    Returns {"days": [date...], "xp": [per-day XP], "knowledge": [course
    knowledge XP at each day's close], "open_knowledge": knowledge at the first
    day's open, "answers": [answers per day], "solved": [answers per day
    that were right]}.

    `also`: concepts outside the course whose K is wanted too (the Learner
    Home's concept list, app/concept_choice.py). They are modelled and
    returned in "also_now" (K) and "also_state" (what `solve_xp` needs); no
    XP total counts them.

    🔴 XP is EARNED over every course; only the course's own numbers
    ("knowledge", "open_knowledge", "today_open_knowledge", "per_kc_now",
    "scope") follow the studied course. Seth, 2026-10-01: switching courses
    "reset my xp on the main leaderboard page" — `scope_kcs` narrowed to
    the studied course, so unticking ARENA dropped every XP point earned on
    it (his state: 175.2 → 0.0). XP is what the learner learned, whichever
    course they study today."""
    now = now or datetime.now(timezone.utc)
    scope = scope_kcs(user_state)
    earn = scope_kcs(None)  # every course: XP never depends on the studied course
    in_scope = set(earn) | set(scope) | set(also)
    start = _prior(user_state)
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
    obs: Dict[str, List[tuple]] = {kc: [] for kc in in_scope}
    seen_exploit: set = set()
    # CREDIT: per concept, the slot of its latest correct answer and the
    # memory just after it.
    last_ok: Dict[str, Tuple[int, memory_model.Memory]] = {}
    snap_R: List[Dict[str, float]] = []
    snap_j: List[Dict[str, int]] = []
    snap_ok: List[Dict[str, Tuple[int, float]]] = []
    answers = [0] * n_days
    # Right answers (a pass, not a miss and not a read answer): the Learner
    # Home's "Problems solved" measure (Seth, 2026-09-28).
    solved = [0] * n_days

    def snapshot(t: float) -> None:
        snap_R.append({kc: memory_model.retrievability(m, t, cfg)
                       for kc, m in mems.items() if kc in in_scope})
        snap_j.append({kc: len(o) for kc, o in obs.items() if o})
        snap_ok.append({kc: (j, memory_model.retrievability(m, t, cfg)) for kc, (j, m) in last_ok.items()})

    def observe(kc: str, t: float, correct: Optional[bool], probe: bool,
                stage: Optional[str] = None, qid: Optional[int] = None) -> None:
        g, s = _likelihood(kc, stage, probe)
        R = memory_model.retrievability(mems[kc], t, cfg) if kc in mems else 1.0
        d = A.difficulty(qid, kc)
        if probe:
            obs[kc].append((None, d, correct, R, g, s, (d, A.RATE_PROBE) if correct else None))
            return
        pre = None
        if kc not in seen_exploit:
            seen_exploit.add(kc)
            pre = (A.lesson_difficulty(kc), A.RATE_LESSON)
        step = (d, A.RATE_AIDED) if correct is None else (d, A.RATE_ANSWER) if correct else None
        obs[kc].append((pre, d, correct, R, g, s, step))

    def credit(kc: str, correct: Optional[bool]) -> None:
        if correct:
            last_ok[kc] = (len(obs[kc]), mems[kc])

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
            if ok and 1 <= bi <= n_days:
                solved[bi - 1] += 1
            if kc in in_scope:
                observe(kc, t, ok, probe=True)
                # The placement answer is a retrieval like any other: without
                # it a concept known only from placement kept R = 1 forever
                # and its starting credit never faded (codex, 2026-09-26).
                # Local to this replay; the app's own memory is unchanged.
                grade = memory_model.GOOD if ok else memory_model.AGAIN
                memory_model._apply(mems, t, {kc: grade}, cfg, None)
                credit(kc, ok)
            continue
        ev = payload
        # One question tagged with several concepts is one event; it was
        # solved when its grades are passes.
        if 1 <= bi <= n_days and ev.grades and all(
                g not in (memory_model.AGAIN, memory_model.AIDED) for g in ev.grades.values()):
            solved[bi - 1] += 1
        outcome = {}
        for kc, grade in ev.grades.items():
            if kc not in in_scope:
                continue
            correct = None if grade == memory_model.AIDED else grade != memory_model.AGAIN
            probe, stage = _meta_for(meta, kc, ev.qid, t)
            observe(kc, t, correct, probe, stage, ev.qid)
            outcome[kc] = correct
        memory_model._apply(mems, t, ev.grades, cfg, ev.p_skill)
        for kc, correct in outcome.items():
            credit(kc, correct)
    while bi < len(bound_t):
        snapshot(bound_t[bi])
        bi += 1

    smoothed = {kc: A.smooth(start, o) for kc, o in obs.items() if o}
    at_start = {kc: A.knowledge(start, kc) for kc in in_scope}

    def post(b: int, kc: str):
        sm = smoothed.get(kc)
        return sm[snap_j[b].get(kc, 0)] if sm else start

    def skill_at(kc: str, j: int) -> float:
        sm = smoothed.get(kc)
        return A.knowledge(sm[j], kc) if sm else at_start[kc]

    def knowledge(b: int, kc: str) -> float:
        return skill_at(kc, snap_j[b].get(kc, 0)) * snap_R[b].get(kc, 1.0)

    def credited(b: int, kc: str) -> float:
        """Knowledge as of the latest correct answer (CREDIT); before any,
        the opening belief, which is starting credit."""
        hit = snap_ok[b].get(kc)
        return skill_at(kc, 0) if hit is None else skill_at(kc, hit[0]) * hit[1]

    # XP per unit of capped K: a concept pays its worth at READY. XP over
    # `earn` (every course), course knowledge over `scope` (studied course).
    def priced(kcs, read):
        return ([worth(kc) / READY for kc in kcs],
                [[min(read(b, kc), READY) for kc in kcs] for b in range(len(bounds))])

    earn_rate, earn_capped = priced(earn, credited)
    rate, capped = priced(scope, knowledge)
    xp, closing = [], []
    for i in range(n_days):
        xp.append(sum(w * max(0.0, c1 - c0)
                      for w, c0, c1 in zip(earn_rate, earn_capped[i], earn_capped[i + 1])))
        closing.append(sum(w * c for w, c in zip(rate, capped[i + 1])))
    last = len(bounds) - 1

    def state(kc):
        return {"post": post(last, kc), "R": snap_R[last].get(kc, 1.0), "lesson_seen": kc in seen_exploit}

    return {
        "scope": scope,
        "days": [first + timedelta(days=i) for i in range(n_days)],
        "xp": xp,
        "knowledge": closing,
        "open_knowledge": sum(w * c for w, c in zip(rate, capped[0])),
        "today_open_knowledge": sum(w * c for w, c in zip(rate, capped[n_days - 1])),
        "answers": answers,
        "solved": solved,
        "per_kc_now": {kc: knowledge(last, kc) for kc in scope},
        "state_now": {kc: state(kc) for kc in scope},
        "also_now": {kc: knowledge(last, kc) for kc in also},
        "also_state": {kc: state(kc) for kc in also},
    }


def solve_xp(kc: str, state: dict) -> float:
    """The XP the model expects from SOLVING the next problem on `kc` (the
    Learner Home's concept list), served at the learner's level
    (`ability_model.at_level`): the lesson first if the concept was never
    practised, then a correct answer — evidence, read through the emission —
    and its practice step, with recall back to 1, as a correct answer is a
    retrieval. A wrong answer pays nothing (CREDIT). `state` is one entry of
    replay's "state_now" / "also_state". An estimate: the smoothed replay that
    pays the XP can move some of it to an earlier day, and the day's floor can
    hold some back."""
    post, R = state["post"], float(state["R"])
    g, s = _likelihood(kc, None, False)
    v = post if state.get("lesson_seen") else post @ A.transition(A.lesson_difficulty(kc), A.RATE_LESSON)
    d = A.at_level(v, kc)
    v = v * A.emission(d, True, R, g, s)
    v = (v / v.sum()) @ A.transition(d, A.RATE_ANSWER)
    before = A.knowledge(post, kc) * R
    return worth(kc) / READY * max(0.0, min(A.knowledge(v, kc), READY) - min(before, READY))


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
    total = sum(worth(kc) for kc in r["scope"])
    earned = sum(r["xp"])
    know_now = r["knowledge"][-1]
    remaining = max(0.0, total - know_now)
    today = r["days"][-1]
    target = getattr(user_state, "xp_target", None) or None
    remaining_open = max(0.0, total - r["today_open_knowledge"])
    need_today = daily_target(target, remaining_open, today)
    pace = _pace(r, zone, now)
    finish = _finish(today, remaining, pace)
    ready = sum(1 for v in r["per_kc_now"].values() if v >= READY - 1e-9)
    from app import learning_pace
    left_problems, left_minutes, away = learning_pace.remaining(r["state_now"])
    in_reach = max(0.0, remaining - sum(worth(kc) * (1.0 - min(r["per_kc_now"].get(kc, 0.0), READY) / READY)
                                        for kc in away))
    return {
        "course": {**course_info(user_state), "concepts": len(r["scope"]), "total_xp": total,
                   "ready_at": READY_PCT, "ready_concepts": ready},
        "earned": round(earned, 1),
        "knowledge": round(know_now, 1),
        "starting_credit": round(r["open_knowledge"], 1),
        "remaining": round(remaining, 1),
        # What a date target divides: the remainder at today's OPEN, so today's
        # own learning does not shrink today's number. The form's preview uses it.
        "remaining_open": round(remaining_open, 1),
        # Problems (and assumed minutes) the remainder takes by the same
        # model, concept by concept (app/learning_pace): the home turns an XP
        # target into "≈ N problems a day" with them, over the XP they cover —
        # concepts whose bank has no problem near the learner are counted in
        # none of the three.
        "problems_remaining": round(left_problems, 1),
        "minutes_remaining": round(left_minutes),
        "remaining_in_reach": round(in_reach, 1),
        "concepts_out_of_reach": len(away),
        "today": {"date": today.isoformat(), "xp": round(r["xp"][-1], 1),
                  "target": need_today, "provisional": True},
        "target": target,
        "pace": round(pace, 1),
        "pace_days": PACE_DAYS,
        "projected_finish": finish,
        "days": [{"date": d.isoformat(), "xp": round(x, 1), "knowledge": round(k, 1), "answers": a,
                  "solved": v}
                 for d, x, k, a, v in zip(r["days"], r["xp"], r["knowledge"], r["answers"], r["solved"])],
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
