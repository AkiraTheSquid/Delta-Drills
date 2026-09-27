"""Stored placement evidence — READ ONLY since 2026-09-26.

Seth retired the placement test ("remove the diagnostic test distinction, but
keep explore/exploit"): nothing starts, serves or records a placement any
more. The Practice tab's quick survey (`area_survey.py`) and the explore
phase (`kc_explore.py`) do its job. What is left here reads what earlier
placements STORED in `UserPracticeState.diagnostic` — their probe logs are
real answers by real learners and keep counting:

  * `beliefs` — per-KC P(known) from the frozen priors + probe log
    (`placement_model`), read by `practice_targets` (ray readiness);
  * `area_estimates` / `display_area_estimates` — the frozen per-area
    readout the group board draws;
  * `can_set_prior` — whether the self-reported level may still move;
  * `kc_cap_secs` — each concept's answer clock
    (lessons/placement_time_caps.json), which ordinary practice uses too;
  * `_arena_links` — the ARENA exercise → concept map.

`get_diag` always reads a run as finished-or-never-started: a run left active
when the test was retired has no page to finish it on, and no path may treat
its learner as mid-placement.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from app import bkt_mastery, kc_graph, kc_prefs, placement_model, practice_targets
from app.adaptive import UserPracticeState

logger = logging.getLogger(__name__)

# --- answer clocks ---------------------------------------------------------------------
# The DEFAULT answer clock, and the ceiling: no concept may be given more.
# The clock a probe actually gets is PER CONCEPT — lessons/placement_time_caps.json
# (kc_cap_secs below), one fixed number per KC for every learner. Seth,
# 2026-09-07: einops and broadcasting get ARENA's own ~10 minutes, the
# one-call py-0/np-1 drills five or six, make_rays_1d fifteen. Per concept,
# never per question or per learner, so probes on one concept stay comparable.
PER_PROBLEM_SECS = 20 * 60


_ARENA_MAP_PATH = kc_graph._LESSONS_DIR / "arena_exercise_kcs.json"
_TIME_CAPS_PATH = kc_graph._LESSONS_DIR / "placement_time_caps.json"


@lru_cache(maxsize=1)
def _time_caps() -> Tuple[int, Dict[str, int]]:
    """(default_secs, {kc: secs}) from placement_time_caps.json. Every value is
    clamped to (0, PER_PROBLEM_SECS]: the file may shorten a concept's clock,
    never lengthen it past the ceiling the client's constant also enforces."""
    raw = kc_graph._read_json(_TIME_CAPS_PATH) or {}

    def _clamp(v) -> Optional[int]:
        if isinstance(v, bool):          # JSON true would be a one-second clock
            return None
        try:
            n = int(v)
        except (TypeError, ValueError):
            return None
        return min(PER_PROBLEM_SECS, n) if n > 0 else None

    default = _clamp(raw.get("default_secs")) or PER_PROBLEM_SECS
    kcs = {}
    for kc, v in (raw.get("kcs") or {}).items():
        n = _clamp(v)
        if n is not None:
            kcs[str(kc)] = n
    return default, kcs


def kc_cap_secs(kc: Optional[str]) -> int:
    """The answer clock for a probe on `kc`, seconds."""
    default, kcs = _time_caps()
    return kcs.get(kc, default) if kc else default


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

    # Placement is retired (2026-09-26): a run left active then can never be
    # finished, and nothing may serve it probes. Read it as not running; its
    # probe log stays exactly where it is and still counts as evidence.
    d["active"] = False
    d["pending"] = None
    return d


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
    """Every registry concept the learner has not switched off, inside the
    stored run's scope (practice_targets) and chosen areas (registry topics)."""
    d = get_diag(user_state)
    scope_set = practice_targets.scope_kcs(d.get("scope", "all"))
    areas = set(d.get("areas") or ()) or None
    reg = kc_graph._registry()
    return [k for k, node in reg.items() if not kc_prefs.is_disabled(user_state, k)
            and (scope_set is None or k in scope_set)
            and (areas is None or str(node.get("topic") or "Other") in areas)]


def _graph(user_state: UserPracticeState) -> placement_model.Graph:
    reg = kc_graph._registry()
    return placement_model.Graph({k: n["prereqs"] for k, n in reg.items()}, assessed_kcs(user_state))


def _priors(user_state: UserPracticeState, kcs: List[str]) -> Dict[str, float]:
    """The run's priors, frozen when it started; computed live only for a
    state that predates the freeze or has no run."""
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


# --- the per-concept readout ------------------------------------------------------------------------


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
        # Fast section placement reports uncertainty for untested concepts.
        # Never seed them (or shared atoms) from sparse indirect evidence.
        if diag.get("scope") == practice_targets.RAY:
            continue
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


# --- the grouped readout (the group board) ----------------------------------------------------------

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


