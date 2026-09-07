"""Placement diagnostic — lifecycle, time budget, item choice, seeding.

2026-09-07 rewrite (Seth): the placement now estimates EVERY concept in the
knowledge graph, not four bank topics. The estimator itself is
`placement_model.py` (per-KC P(known) on the prerequisite DAG, ALEKS-style
asymmetric updates, value-weighted selection); this module owns everything
around it:

  * the learner's TIME plan — 1h / 3h / 6h of problem time, every problem
    capped at PER_PROBLEM_SECS. Time, not a question count, is the budget:
    "so that it has a total of 3 hours that is allotted and it can't go
    beyond that maximum, and then I'll have 20 minutes per problem".
    Spent time is measured server-side from serve to answer (capped), so a
    paused tab cannot buy extra questions and a reload cannot lose the count.
  * disabled concepts (graph Settings tab, kc_prefs) are OUT of the test;
    the learner's priority weights scale which concepts get probed first.
  * ARENA coverage: a concept that ARENA's own exercises test
    (lessons/arena_exercise_kcs.json) is probed with one of THOSE problems
    before any generic drill — "a full audit of my preparedness for the
    ARENA curriculum itself".
  * finish: per-concept beliefs seed the atom-BKT priors through the
    crosswalk; uncertain concepts are flagged for fast-tracking; every
    prerequisite edge the learner's evidence contradicts is reported.

Public names are unchanged from the topic-area version on purpose:
`questions_router.py` (another session's file) calls should_run /
should_finish / select_probe / record_probe / override_probe / finish /
get_diag / effective_budget, and keeps working untouched.

State lives in `UserPracticeState.diagnostic`, a plain JSON dict; beliefs are
recomputed from the probe log on demand, so nothing derived can drift.
"""

from __future__ import annotations

import json
import logging
import statistics
from datetime import datetime, timezone
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from app import bkt_mastery, kc_graph, kc_prefs, placement_model
from app.adaptive import UserPracticeState
from app.questions import get_question_by_id, get_subtopics, get_topic_for_subtopic

logger = logging.getLogger(__name__)

# --- the learner's choices ----------------------------------------------------------
PLAN_HOURS: Tuple[int, ...] = (1, 3, 6)
DEFAULT_PLAN_HOURS = 1
PER_PROBLEM_SECS = 20 * 60
# A test ends when less than this much problem time is left: a 40-second
# remainder cannot hold a problem.
MIN_REMAINING_SECS = 60

# A KC stops being worth a probe when the best single probe would remove less
# than this much variance from the KC itself (unweighted). Roughly: two
# agreeing direct probes settle a concept; disagreement keeps it open.
STOP_GAIN = 0.01
# Coverage-first: a concept ARENA tests, not yet probed with an ARENA problem,
# sorts as if it were this much more valuable. Audit before fine-tuning.
ARENA_BONUS = 1.5

# Seeding: a placement may unlock (>0.85) but never certify mastery.
SEED_MASTERY_FLOOR = 0.02
SEED_MASTERY_CAP = 0.92

# Time-estimate heuristic per problem, seconds, from the bank difficulty
# (0-100): a difficulty-20 drill ≈ 3 min, 60 ≈ 6 min, 100 ≈ 9 min. v0 — no
# learner timing exists yet (latency was never logged); replace with the
# per-probe `secs` this module now records once there is a run to read.
EST_BASE_SECS = 90.0
EST_SECS_PER_DIFFICULTY = 4.5

_ARENA_MAP_PATH = kc_graph._LESSONS_DIR / "arena_exercise_kcs.json"


# --- state accessors -------------------------------------------------------------------


def get_diag(user_state: UserPracticeState) -> dict:
    """The user's diagnostic dict (created empty on first touch)."""
    if not isinstance(getattr(user_state, "diagnostic", None), dict):
        user_state.diagnostic = {}
    d = user_state.diagnostic
    d.setdefault("active", False)
    d.setdefault("completed_at", None)
    d.setdefault("declined", False)
    d.setdefault("probes", [])
    d.setdefault("plan", None)
    d.setdefault("spent_secs", 0)
    d.setdefault("pending", None)
    return d


def should_run(user_state: UserPracticeState) -> bool:
    """True only after an explicit start from the placement page."""
    return bool(get_diag(user_state)["active"])


def can_set_prior(user_state: UserPracticeState) -> bool:
    """Whether self-report can still be an honest cold-start prior: nothing
    has been answered or probed yet."""
    d = get_diag(user_state)
    if d["completed_at"] or d["probes"]:
        return False
    if getattr(user_state, "atom_mastery", None):
        return False
    if any(s.n > 0 or s.history for s in user_state.subtopic_states.values()):
        return False
    for features in getattr(user_state, "kc_posteriors", {}).values():
        if any(int(p.get("n") or 0) > 0 for p in features.values() if isinstance(p, dict)):
            return False
    return True


def normalize_hours(hours) -> int:
    try:
        h = int(hours)
    except (TypeError, ValueError):
        return DEFAULT_PLAN_HOURS
    return h if h in PLAN_HOURS else DEFAULT_PLAN_HOURS


def start(user_state: UserPracticeState, hours=None) -> dict:
    """Explicitly (re)start: clears the probe log and the clock, keeps BKT.
    The previous run's frozen estimates are dropped with it — a retake is a
    new measurement, not an amendment."""
    d = get_diag(user_state)
    h = normalize_hours(hours)
    d["active"] = True
    d["declined"] = False
    d["completed_at"] = None
    d["probes"] = []
    d["plan"] = {
        "hours": h,
        "budget_secs": h * 3600,
        "per_problem_secs": PER_PROBLEM_SECS,
        "started_at": _now().isoformat(),
    }
    d["spent_secs"] = 0
    d["pending"] = None
    for key in ("estimates", "kcs", "atoms_seeded", "atoms_written", "fast_track", "edge_violations"):
        d.pop(key, None)
    # 🔴 Priors are FROZEN here, before any seeding. finish() writes placement
    # numbers into atom_mastery, and a prior recomputed from those would feed
    # the run's own output back in as evidence — a second finish() (retry,
    # /override re-seed) then amplified every belief. Frozen, the posterior
    # is a pure function of the log.
    d["priors"] = _priors_from_history(user_state, assessed_kcs(user_state))
    return d


def decline(user_state: UserPracticeState) -> dict:
    """Opt out — never auto-start again, no seeding."""
    d = get_diag(user_state)
    d["active"] = False
    d["declined"] = True
    d["pending"] = None
    return d


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --- the graph under assessment ------------------------------------------------------------


@lru_cache(maxsize=1)
def _arena_links() -> Dict[str, dict]:
    """KC -> {"question_ids": set, "exercises": [fn...], "notebooks": [slug...]}
    from lessons/arena_exercise_kcs.json — the problems ARENA itself poses."""
    try:
        raw = json.loads(_ARENA_MAP_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("diagnostic: no ARENA exercise map at %s", _ARENA_MAP_PATH)
        return {}
    out: Dict[str, dict] = {}
    for slug, exercises in raw.items():
        if slug.startswith("_") or not isinstance(exercises, dict):
            continue
        for fn, row in exercises.items():
            if not isinstance(row, dict) or not row.get("kc"):
                continue
            entry = out.setdefault(row["kc"], {"question_ids": set(), "exercises": [], "notebooks": []})
            ids = [row.get("original")] + list(row.get("variants") or [])
            entry["question_ids"].update(int(i) for i in ids if isinstance(i, int))
            entry["exercises"].append(fn)
            if slug not in entry["notebooks"]:
                entry["notebooks"].append(slug)
    return out


def reload_caches() -> None:
    _arena_links.cache_clear()


def assessed_kcs(user_state: UserPracticeState) -> List[str]:
    """Every registry concept the learner has not switched off."""
    return [k for k in kc_graph._registry() if not kc_prefs.is_disabled(user_state, k)]


def _graph(user_state: UserPracticeState) -> placement_model.Graph:
    reg = kc_graph._registry()
    return placement_model.Graph({k: n["prereqs"] for k, n in reg.items()}, assessed_kcs(user_state))


def _values(user_state: UserPracticeState, kcs: List[str]) -> Dict[str, float]:
    """Coreness x the learner's own weight — what a concept's certainty is worth."""
    descendants, _ = kc_graph._closure()
    return {k: (descendants.get(k, 0) + 1) * (kc_prefs.weight_for(user_state, k) or 1.0) for k in kcs}


def _priors(user_state: UserPracticeState, kcs: List[str]) -> Dict[str, float]:
    """The run's frozen priors (see start()); computed live only for a state
    that predates the freeze or has no run yet."""
    frozen = get_diag(user_state).get("priors")
    if isinstance(frozen, dict) and frozen:
        level = getattr(user_state, "self_reported_level", None)
        return {
            k: float(frozen[k]) if k in frozen else placement_model.prior_logodds(None, level)
            for k in kcs
        }
    return _priors_from_history(user_state, kcs)


def _priors_from_history(user_state: UserPracticeState, kcs: List[str]) -> Dict[str, float]:
    """Prior log-odds per KC from the learner's own atom-BKT history — the
    crosswalk-weighted mean over the KC's atoms that have actually been
    practised. 🔴 Not `kc_graph.kc_mastery`: that averages every atom, with the
    never-practised ones sitting at p_init (0.10), so a concept with one strong
    atom and five untouched ones read as 0.15 and started every run "unknown".
    A KC with no practised atom has no evidence and sits at 0.5 (+ the
    self-report nudge)."""
    level = getattr(user_state, "self_reported_level", None)
    params = bkt_mastery.params_for_level(level)
    now = _now()
    mastery = getattr(user_state, "atom_mastery", None) or {}
    last_ts = getattr(user_state, "atom_last_ts", None) or {}
    out: Dict[str, float] = {}
    for kc in kcs:
        row = kc_graph.crosswalk_row(kc) or {}
        acc = total = 0.0
        for a in row.get("atoms") or []:
            atom = a.get("a")
            w = float(a.get("w") or 0.0)
            if not atom or w <= 0 or atom not in mastery:
                continue
            acc += w * bkt_mastery.current_mastery(mastery, last_ts, atom, now, params)
            total += w
        out[kc] = placement_model.prior_logodds(acc / total if total > 0 else None, level)
    return out


def _model_probes(diag: dict) -> List[dict]:
    return [p for p in diag["probes"] if p.get("kc")]


def beliefs(user_state: UserPracticeState) -> Dict[str, float]:
    """P(known) for every assessed KC, from prior + probe log."""
    kcs = assessed_kcs(user_state)
    if not kcs:
        return {}
    return placement_model.posterior(_priors(user_state, kcs), _graph(user_state), _model_probes(get_diag(user_state)))


# --- time --------------------------------------------------------------------------------------


def est_secs_for_question(q) -> float:
    return EST_BASE_SECS + EST_SECS_PER_DIFFICULTY * float(getattr(q, "difficulty_score", 50) or 50)


def _mean_est_secs(kcs: List[str]) -> float:
    ests = []
    for kc in kcs:
        for qid in kc_graph.questions_for_kc(kc):
            q = get_question_by_id(qid)
            if q is not None:
                ests.append(est_secs_for_question(q))
    return statistics.fmean(ests) if ests else 300.0


def remaining_secs(user_state: UserPracticeState) -> int:
    d = get_diag(user_state)
    plan = d.get("plan") or {}
    budget = int(plan.get("budget_secs") or DEFAULT_PLAN_HOURS * 3600)
    return max(0, budget - int(d.get("spent_secs") or 0))


def plan_options(user_state: UserPracticeState) -> List[dict]:
    """The picker: for each allowed length, how many problems it is likely to
    hold and how long the test will probably actually take. Point estimates
    from the difficulty heuristic — 20 minutes is the CAP per problem, not the
    expectation, so a 3h plan of ~9 capped problems is really ~35 problems."""
    kcs = assessed_kcs(user_state)
    mean_secs = _mean_est_secs(kcs)
    # Roughly two direct probes settle a concept (STOP_GAIN), so this is the
    # size of a test that finishes on evidence rather than on the clock.
    probes_to_settle = max(1, 2 * len(kcs))
    out = []
    for h in PLAN_HOURS:
        budget = h * 3600
        fits = int(budget // mean_secs) if mean_secs > 0 else probes_to_settle
        probes = max(1, min(fits, probes_to_settle))
        out.append({
            "hours": h,
            "budget_secs": budget,
            "per_problem_secs": PER_PROBLEM_SECS,
            "est_probes": probes,
            "est_minutes": int(round(min(budget, probes * mean_secs) / 60.0)),
            "min_probes_at_cap": max(1, budget // PER_PROBLEM_SECS),
        })
    return out


def effective_budget(user_state: UserPracticeState) -> int:
    """Estimated total problems this run will hold (done + what the remaining
    time is likely to fit). A ceiling for the progress bar, not a promise."""
    d = get_diag(user_state)
    done = len(d["probes"])
    if d["completed_at"]:
        return max(1, done)
    mean_secs = _mean_est_secs(assessed_kcs(user_state))
    fits = int(remaining_secs(user_state) // mean_secs) if mean_secs > 0 else 0
    return max(1, done + fits)


def effective_min_probes(user_state: UserPracticeState) -> int:
    """Earliest a run can finish on evidence: one problem per still-uncertain
    concept, floored at 1."""
    P = beliefs(user_state)
    open_kcs = sum(1 for p in P.values() if placement_model.classify(p) == "uncertain")
    return max(1, len(get_diag(user_state)["probes"]) + open_kcs)


# --- probe selection -------------------------------------------------------------------------------


def _served_ids(user_state: UserPracticeState) -> set:
    return kc_graph._served_question_ids(user_state) | {p["question_id"] for p in get_diag(user_state)["probes"]}


def _arena_probed(diag: dict) -> set:
    return {p["kc"] for p in diag["probes"] if p.get("arena") and p.get("kc")}


def _candidates(user_state: UserPracticeState, kcs: List[str]) -> Dict[str, Tuple[List[int], List[int]]]:
    """KC -> (unserved ARENA-linked question ids, unserved generic ids)."""
    served = _served_ids(user_state)
    links = _arena_links()
    out: Dict[str, Tuple[List[int], List[int]]] = {}
    for kc in kcs:
        pool = [q for q in kc_graph.questions_for_kc(kc) if q not in served]
        if not pool:
            continue
        arena_ids = links.get(kc, {}).get("question_ids", set())
        arena = [q for q in pool if q in arena_ids]
        generic = [q for q in pool if q not in arena_ids]
        out[kc] = (arena, generic)
    return out


def _pick_question(kc: str, arena: List[int], generic: List[int]):
    """ARENA's own problem first; otherwise the drill nearest the pool's median
    difficulty (the item that says most about a binary known/unknown)."""
    for qid in sorted(arena):
        q = get_question_by_id(qid)
        if q is not None:
            return q
    qs = [q for q in (get_question_by_id(i) for i in generic) if q is not None]
    if not qs:
        return None
    median = statistics.median(float(q.difficulty_score or 50) for q in qs)
    qs.sort(key=lambda q: (abs(float(q.difficulty_score or 50) - median), q.id))
    return qs[0]


def _ranked(user_state: UserPracticeState):
    """(ranked, cands, graph, beliefs): (score, kc) best-first over KCs that
    still have an unserved question, the candidate table, and the evaluation
    context so the stop rule need not rebuild it. ARENA-untested concepts get
    the coverage bonus."""
    kcs = assessed_kcs(user_state)
    if not kcs:
        return [], {}, None, None
    diag = get_diag(user_state)
    graph = _graph(user_state)
    B = placement_model.beliefs_from(_priors(user_state, kcs), graph, _model_probes(diag))
    cands = _candidates(user_state, kcs)
    if not cands:
        return [], cands, graph, B
    values = _values(user_state, kcs)
    arena_done = _arena_probed(diag)
    for kc, (arena, _generic) in cands.items():
        if arena and kc not in arena_done:
            values[kc] *= ARENA_BONUS
    ranked = placement_model.rank_beliefs(B, graph, values, list(cands))
    return ranked, cands, graph, B


def select_probe(user_state: UserPracticeState):
    """The next placement problem, or None when nothing informative is left."""
    ranked, cands, _graph_, _B = _ranked(user_state)
    for _score, kc in ranked:
        arena, generic = cands[kc]
        q = _pick_question(kc, arena, generic)
        if q is not None:
            d = get_diag(user_state)
            d["pending"] = {"question_id": q.id, "kc": kc, "served_at": _now().isoformat()}
            return q
    return None


# --- recording + stopping -----------------------------------------------------------------------------


def problem_secs_allowed(user_state: UserPracticeState) -> int:
    """The clock the NEXT problem gets: the per-problem cap, or whatever is
    left of the plan if that is shorter. The last problem of a run is the
    only one that can be short — this is what keeps a 3h plan from ending at
    3h19m (codex, first review)."""
    d = get_diag(user_state)
    cap = int((d.get("plan") or {}).get("per_problem_secs") or PER_PROBLEM_SECS)
    return max(0, min(cap, remaining_secs(user_state)))


def _elapsed_for(user_state: UserPracticeState, question_id: int, elapsed_secs: Optional[float]) -> int:
    """Problem time to charge. 🔴 THE SERVER CLOCK IS AUTHORITATIVE: when the
    answered question is the one served, serve-to-answer is charged and the
    client's own reading is ignored — a client that reports 0 for every
    answer would otherwise never run out of time. The client's number is used
    only when the server has nothing (an answer to a question it did not
    serve, or a pre-rewrite state). Capped at what this problem was allowed,
    never negative."""
    diag = get_diag(user_state)
    cap = problem_secs_allowed(user_state)
    pending = diag.get("pending") or {}
    secs: Optional[float] = None
    if pending.get("question_id") == question_id and pending.get("served_at"):
        try:
            secs = (_now() - datetime.fromisoformat(pending["served_at"])).total_seconds()
        except (TypeError, ValueError):
            secs = None
    if secs is None and elapsed_secs is not None:
        try:
            secs = float(elapsed_secs)
        except (TypeError, ValueError):
            secs = None
    if secs is None:
        secs = 0.0
    return int(max(0.0, min(float(cap), secs)))


def _kc_for(question) -> Optional[str]:
    """The concept a probe is evidence about: its first assessed target KC."""
    kcs = kc_graph.question_kcs(question.id)
    return kcs[0] if kcs else None


def record_probe(
    user_state: UserPracticeState,
    question,
    result: str,
    elapsed_secs: Optional[float] = None,
) -> dict:
    """Log one response ("correct" | "incorrect" | "dont_know"), charge its
    time, then auto-finish if the stopping rule fires."""
    diag = get_diag(user_state)
    if diag["completed_at"]:
        return diag
    secs = _elapsed_for(user_state, question.id, elapsed_secs)
    kc = _kc_for(question)
    pending = diag.get("pending") or {}
    if pending.get("question_id") == question.id and pending.get("kc"):
        kc = pending["kc"]
    arena = kc is not None and question.id in _arena_links().get(kc, {}).get("question_ids", set())

    # A re-answer REPLACES the earlier record, time included — a retried
    # request must not be charged twice.
    for old in diag["probes"]:
        if old["question_id"] == question.id:
            diag["spent_secs"] = max(0, int(diag.get("spent_secs") or 0) - int(old.get("secs") or 0))
    diag["probes"] = [p for p in diag["probes"] if p["question_id"] != question.id]
    diag["probes"].append({
        "question_id": question.id,
        "kc": kc,
        "subtopic": question.subtopic,
        "topic": question.topic or "Other",
        "difficulty": question.difficulty_score,
        "result": result,
        "secs": secs,
        "arena": bool(arena),
        "ts": _now().isoformat(),
    })
    diag["spent_secs"] = int(diag.get("spent_secs") or 0) + secs
    diag["pending"] = None
    if _should_stop(user_state):
        finish(user_state)
    return diag


def override_probe(user_state: UserPracticeState, question_id: int, correct: bool) -> bool:
    """Flip the most recent probe's result (the /override path during
    placement). A run already closed by that probe is re-finished from the
    corrected log here — `finish()` itself is idempotent, so the router's own
    follow-up call is a no-op."""
    diag = get_diag(user_state)
    if not diag["probes"] or diag["probes"][-1]["question_id"] != question_id:
        return False
    diag["probes"][-1]["result"] = "correct" if correct else "incorrect"
    if diag["completed_at"]:
        finish(user_state, refinish=True)
    return True


def should_finish(user_state: UserPracticeState) -> bool:
    """Public stop-check for serving paths — re-checked BEFORE a probe is
    selected, not only after one is recorded."""
    return _should_stop(user_state)


def _should_stop(user_state: UserPracticeState) -> bool:
    if remaining_secs(user_state) < MIN_REMAINING_SECS:
        return True
    ranked, cands, graph, B = _ranked(user_state)
    if not ranked:
        return True
    # Coverage floor: ARENA-tested concepts still awaiting their ARENA problem
    # keep the test open regardless of how settled the graph looks.
    arena_done = _arena_probed(get_diag(user_state))
    if any(arena and kc not in arena_done for kc, (arena, _g) in cands.items()):
        return False
    # Evidence stop: no concept left that its OWN probe would move —
    # `{kc: 1.0}` weighs that concept alone (weighted_variance gives absent
    # KCs weight 0), so propagation elsewhere neither prolongs nor cuts short.
    best = 0.0
    for kc in cands:
        best = max(best, placement_model.expected_variance_reduction(B, graph, kc, {kc: 1.0}))
    return best < STOP_GAIN


# --- finish: seed BKT + report -----------------------------------------------------------------------


def _kc_rows(user_state: UserPracticeState) -> List[dict]:
    diag = get_diag(user_state)
    P = beliefs(user_state)
    reg = kc_graph._registry()
    links = _arena_links()
    counts: Dict[str, int] = {}
    arena_counts: Dict[str, int] = {}
    for p in diag["probes"]:
        if p.get("kc"):
            counts[p["kc"]] = counts.get(p["kc"], 0) + 1
            if p.get("arena"):
                arena_counts[p["kc"]] = arena_counts.get(p["kc"], 0) + 1
    rows = []
    for kc, p in P.items():
        node = reg.get(kc) or {}
        rows.append({
            "kc": kc,
            "title": node.get("title") or kc,
            "lesson": node.get("lesson"),
            "topic": node.get("topic"),
            "p": round(p, 3),
            "state": placement_model.classify(p),
            "probes": counts.get(kc, 0),
            "arena_probes": arena_counts.get(kc, 0),
            "arena_linked": bool(links.get(kc)),
        })
    rows.sort(key=lambda r: r["kc"])
    return rows


def kc_estimates(user_state: UserPracticeState) -> List[dict]:
    """Per-concept readout. A COMPLETED run returns the rows frozen at
    finish() — later practice must not rewrite a finished placement."""
    diag = get_diag(user_state)
    if diag["completed_at"] and diag.get("kcs"):
        return diag["kcs"]
    return _kc_rows(user_state)


def _mastery_from_p(p: float) -> float:
    return max(SEED_MASTERY_FLOOR, min(SEED_MASTERY_CAP, p))


def finish(user_state: UserPracticeState, refinish: bool = False) -> dict:
    """Close the run and seed per-atom BKT mastery from the per-concept beliefs
    through the crosswalk. An atom with RECENT practice evidence (inside the
    BKT half-life) only ever rises; an atom with none — never practised, or
    practised so long ago the decay model has written it off — takes the
    placement's number outright, in either direction. So a self-reported
    "strong" who bombs placement is corrected down, and a stale high number
    from July cannot survive a miss today.

    IDEMPOTENT: a second call on a closed run returns it unchanged (the last
    answer auto-finishes, then the client's own /finish lands). Only
    `override_probe` re-finishes, and it says so with `refinish`; the seeding
    then re-applies from the SAME frozen priors and the corrected log, and
    atoms this run already wrote are simply overwritten with the new number."""
    diag = get_diag(user_state)
    if diag["completed_at"] and not refinish:
        return diag
    now = _now()
    rows = _kc_rows(user_state)
    P = {r["kc"]: r["p"] for r in rows}

    params = bkt_mastery.params_for_level(user_state.self_reported_level)
    seeded: Dict[str, float] = {}
    for kc, p in P.items():
        row = kc_graph.crosswalk_row(kc) or {}
        m = _mastery_from_p(p)
        for a in row.get("atoms") or []:
            atom = a.get("a")
            if not atom or float(a.get("w") or 0.0) <= 0:
                continue
            if atom not in seeded or m > seeded[atom]:
                seeded[atom] = m

    applied = 0
    written_before = set(diag.get("atoms_written") or [])
    for atom, m in seeded.items():
        if atom in user_state.atom_mastery and atom not in written_before:
            recent = _is_recent(user_state.atom_last_ts.get(atom), now)
            current = bkt_mastery.current_mastery(
                user_state.atom_mastery, user_state.atom_last_ts, atom, now, params
            )
            if recent and m <= current:
                continue
        user_state.atom_mastery[atom] = m
        user_state.atom_last_ts[atom] = now.isoformat()
        applied += 1

    graph = _graph(user_state)
    diag["active"] = False
    diag["pending"] = None
    diag["completed_at"] = now.isoformat()
    diag["kcs"] = rows
    diag["estimates"] = _area_rows_from(rows)
    diag["atoms_seeded"] = applied
    diag["atoms_written"] = sorted(seeded)
    diag["fast_track"] = [r["kc"] for r in rows if r["state"] == "uncertain"]
    diag["edge_violations"] = placement_model.edge_violations(P, graph)
    return diag


def _is_recent(ts: Optional[str], now: datetime) -> bool:
    if not ts:
        return False
    try:
        age = (now - datetime.fromisoformat(ts)).total_seconds() / 86400.0
    except (TypeError, ValueError):
        return False
    return age <= bkt_mastery.HALF_LIFE_DAYS


def fast_track_kcs(user_state: UserPracticeState) -> List[str]:
    """Concepts the last finished placement left uncertain — the learning mode
    should serve these first (ALEKS: uncertain items are fast-tracked)."""
    d = get_diag(user_state)
    return list(d.get("fast_track") or []) if d["completed_at"] else []


# --- the grouped readout (kept for the existing results card) ---------------------------------------

DISPLAY_AREAS: Tuple[str, ...] = ("PyTorch", "Einops", "Einsum")


def display_area_for_kc(kc: str, topic: Optional[str]) -> str:
    if kc.startswith("einops.einsum"):
        return "Einsum"
    if kc.startswith("einops."):
        return "Einops"
    name = (topic or "").strip()
    return name if name in ("Einops", "Einsum") else "PyTorch"


def _area_rows_from(rows: List[dict]) -> List[dict]:
    """Per-KC rows folded into the three areas the results card draws, in the
    {topic, theta, sd, probes} shape it already reads. theta = mean P(known)
    on 0-100; sd = mean per-concept Bernoulli SD on the same scale — a group
    is no better known than its members are."""
    groups: Dict[str, Dict[str, float]] = {}
    for r in rows:
        name = display_area_for_kc(r["kc"], r.get("topic"))
        g = groups.setdefault(name, {"n": 0, "p": 0.0, "sd": 0.0, "probes": 0})
        p = float(r["p"])
        g["n"] += 1
        g["p"] += p
        g["sd"] += (p * (1.0 - p)) ** 0.5
        g["probes"] += int(r.get("probes") or 0)
    out = []
    for name in DISPLAY_AREAS:
        g = groups.get(name)
        if not g or g["n"] <= 0:
            continue
        out.append({
            "topic": name,
            "theta": round(100.0 * g["p"] / g["n"], 1),
            "sd": round(100.0 * g["sd"] / g["n"], 1),
            "probes": int(g["probes"]),
        })
    return out


def area_estimates(user_state: UserPracticeState) -> List[dict]:
    diag = get_diag(user_state)
    if diag["completed_at"] and diag.get("estimates"):
        return diag["estimates"]
    return _area_rows_from(_kc_rows(user_state))


def display_area_estimates(user_state: UserPracticeState) -> List[dict]:
    return area_estimates(user_state)


# Bank-topic helpers kept importable for anything that still asks.
def _areas() -> List[str]:
    seen: Dict[str, None] = {}
    for st in get_subtopics():
        seen.setdefault(get_topic_for_subtopic(st) or "Other", None)
    return list(seen)
