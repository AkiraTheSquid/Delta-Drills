"""KC-lattice prerequisite gating — the knowledge graph as the ordering authority.

Why this module exists
----------------------
Selection used to run on three disjoint id vocabularies at once:

  * `question_is_unlocked` gated on BKT **atom** prerequisites and returned True
    for any untagged question,
  * `select_next_subtopic` ranked **subtopics** weakest-first, and
  * the knowledge graph drew **KC** prerequisite edges that nothing consulted.

With no evidence every subtopic ties at the BKT prior, so the tiebreak decided,
and the tiebreak was "whichever subtopic has the easiest entry question". That is
how a learner with zero attempts got served cumulative sums before ever seeing
what a tensor is. The graph was right; nothing read it.

This module makes the KC lattice in `lessons/kc_registry.json` the authority,
following the model in the primary sources (`ITS-procedural-AI-SYNC/math-academy`,
The Math Academy Way ch. 4 "Core Technology: the Knowledge Graph" and ch. 32
"Prioritizing Core Topics"):

  * **Knowledge frontier** (ch. 4) — the boundary between what the learner knows
    and what they don't. A KC is *unlocked* only when every prerequisite is
    learned; new work is always served from the frontier, never past it.
  * **Mastery learning** (ch. 4) — prerequisites must be demonstrated before
    more advanced topics become available. The gate is a hard gate, not a
    weighting.
  * **Core topics first** (ch. 32) — among topics on the frontier, prefer the
    ones that "appear more frequently as prerequisites of other topics". We
    operationalise coreness as the transitive descendant count in the
    prerequisite DAG: the more of the course a KC unlocks, the earlier it runs.
  * **No topics are skipped** (ch. 32) — coreness changes the *order*, never the
    membership. Every KC stays reachable.

What this module deliberately does NOT claim
--------------------------------------------
Per-KC mastery is read through `kc_atom_crosswalk.json`, which joins KC ids to
BKT atom ids through the shared question bank. Only 20 of 63 KCs clear the
crosswalk's reliability bar (`tier == "measured"`); the other 43 are
`topic-proxy` — their number is a topic average wearing a per-node label. Gating
uses both, because a proxy beats no signal for ordering, but every estimate
carries its tier out to the API so the graph can draw the difference instead of
implying a measurement that was never taken.

The papers under `papers/prereq-graph-methods/` (RefD, Pan MOOC-prereq, AKD,
ESCO, CLLMRec) are about *inferring* prerequisite edges from text; our edges are
hand-authored in the registry, so they bear on how the lattice was built rather
than how it is served. The one that bears on serving is Hu & Pan's PDRS
cold-start result — prerequisite context helps most when interaction data is
thin, which is exactly the zero-evidence case this module fixes.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app import lessons
from app import example_schedule

from app import bkt_mastery

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LESSONS_DIR = _REPO_ROOT / "Local_Deployed_Shared" / "lessons"
_GRAPH_DIR = _REPO_ROOT / "Local_Deployed_Shared" / "concept-graph"

_REGISTRY_PATH = _LESSONS_DIR / "kc_registry.json"
_QMATRIX_PATH = _LESSONS_DIR / "qmatrix_tags.json"
_CROSSWALK_PATH = _GRAPH_DIR / "kc_atom_crosswalk.json"

# A KC counts as learned — and therefore stops blocking its children and leaves
# the frontier — at the same belief that clears an atom for gating. Using one
# threshold for both keeps "unlocked for my children" and "done as new work"
# from disagreeing, which would otherwise leave a KC being served forever while
# its dependents were already open.
LEARNED_THRESHOLD = bkt_mastery.UNLOCK_THRESHOLD

# Fraction of a KC's crosswalk weight that must sit on atoms the learner has
# actually attempted before the estimate is called evidence-backed. Below this
# the number is still used for ordering (it is the prior, which is the correct
# thing to order on) but is reported as unevidenced.
MIN_COVERED_W = 0.5


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("kc_graph: missing %s — lattice gating disabled", path)
        return {}
    except Exception:
        logger.exception("kc_graph: unreadable %s — lattice gating disabled", path)
        return {}


@lru_cache(maxsize=1)
def _registry() -> Dict[str, dict]:
    """KC id -> {title, lesson, prereqs, subtopic_key, topic}."""
    raw = _read_json(_REGISTRY_PATH)
    lessons = {l.get("id"): l for l in raw.get("lessons", []) if isinstance(l, dict)}
    out: Dict[str, dict] = {}
    for kc in raw.get("kcs", []):
        if not isinstance(kc, dict) or not kc.get("id"):
            continue
        lesson = lessons.get(kc.get("lesson")) or {}
        out[kc["id"]] = {
            "id": kc["id"],
            "title": kc.get("title") or kc["id"],
            "lesson": kc.get("lesson"),
            "lesson_title": lesson.get("title"),
            "topic": lesson.get("topic"),
            "subtopic_key": lesson.get("subtopic_key"),
            # Only prereqs that name a real KC — a typo in the registry must not
            # silently lock a node forever with an unsatisfiable dependency.
            "prereqs": [p for p in (kc.get("prereqs") or []) if isinstance(p, str)],
        }
    for kc in out.values():
        kc["prereqs"] = [p for p in kc["prereqs"] if p in out]
    return out


@lru_cache(maxsize=1)
def _qmatrix() -> Dict[int, dict]:
    """Question id -> {target_kcs, supporting_kcs, source}.

    `source` records WHICH list of a KP's frontmatter claimed the question —
    faded / guided / independent — which is the scaffolding ladder `ladder_rank`
    orders by. It is carried through verbatim; an unrecognised value sorts last
    rather than being silently treated as a rung.
    """
    raw = _read_json(_QMATRIX_PATH)
    out: Dict[int, dict] = {}
    for qid, row in raw.items():
        if not isinstance(row, dict):
            continue
        try:
            key = int(qid)
        except (TypeError, ValueError):
            continue
        out[key] = {
            "target_kcs": [k for k in (row.get("target_kcs") or []) if isinstance(k, str)],
            "supporting_kcs": [k for k in (row.get("supporting_kcs") or []) if isinstance(k, str)],
            "source": row.get("source") if isinstance(row.get("source"), str) else None,
        }
    return out


@lru_cache(maxsize=1)
def _crosswalk() -> Dict[str, dict]:
    """KC id -> crosswalk row (atoms with weights, reliability tier)."""
    raw = _read_json(_CROSSWALK_PATH)
    rows = raw.get("kcs") if isinstance(raw.get("kcs"), dict) else raw
    return {k: v for k, v in rows.items() if isinstance(v, dict) and v.get("atoms")}


@lru_cache(maxsize=1)
def _questions_by_kc() -> Dict[str, Tuple[int, ...]]:
    """KC id -> question ids that TARGET it. Supporting tags are deliberately
    excluded: a question that merely leans on a KC is not practice of it, and
    treating it as such would let the queue claim coverage it never gave."""
    out: Dict[str, List[int]] = {}
    for qid, row in _qmatrix().items():
        for kc in row["target_kcs"]:
            out.setdefault(kc, []).append(qid)
    return {k: tuple(sorted(v)) for k, v in out.items()}


@lru_cache(maxsize=1)
def _closure() -> Tuple[Dict[str, int], Dict[str, int]]:
    """(descendant_count, depth) per KC.

    `descendant_count` is coreness in the sense of Math Academy ch. 32: how many
    KCs transitively require this one. `depth` is the longest prerequisite chain
    ending at the KC, used only to break coreness ties toward the foundations.

    Both are computed with an explicit visited set so a malformed registry that
    contains a prerequisite cycle degrades to a finite answer instead of
    recursing forever.
    """
    reg = _registry()
    children: Dict[str, List[str]] = {k: [] for k in reg}
    for kc, node in reg.items():
        for p in node["prereqs"]:
            children[p].append(kc)

    def reachable(start: str, adj: Dict[str, List[str]]) -> Set[str]:
        seen: Set[str] = set()
        stack = list(adj.get(start, ()))
        while stack:
            n = stack.pop()
            if n in seen or n == start:
                continue
            seen.add(n)
            stack.extend(adj.get(n, ()))
        return seen

    parents = {k: node["prereqs"] for k, node in reg.items()}
    descendants = {k: len(reachable(k, children)) for k in reg}

    depth: Dict[str, int] = {}

    def compute_depth(kc: str, path: Set[str]) -> int:
        if kc in depth:
            return depth[kc]
        if kc in path:  # cycle — stop rather than recurse
            return 0
        ps = parents.get(kc) or []
        d = 0 if not ps else 1 + max(compute_depth(p, path | {kc}) for p in ps)
        depth[kc] = d
        return d

    for kc in reg:
        compute_depth(kc, set())
    return descendants, depth


def reload_caches() -> None:
    """Drop the cached registry/q-matrix/crosswalk. For tests and for the
    content pipeline, which rewrites these files on export."""
    for fn in (_registry, _qmatrix, _crosswalk, _questions_by_kc, _closure):
        fn.cache_clear()


# --- learner-facing state -----------------------------------------------------


def kc_mastery(user_state, kc: str) -> Tuple[float, float, str]:
    """(mastery, covered_weight, tier) for one KC.

    Mastery is the crosswalk-weighted mean of the learner's decay-adjusted BKT
    posteriors over the atoms the KC's questions exercise. Atoms with no
    attempts sit at the learner's prior, so a fresh account lands near the BKT
    prior everywhere — which is the honest answer and the one that makes the
    whole lattice lock except at its roots.

    `covered_weight` is the share of the KC's crosswalk weight that rests on
    atoms the learner has actually attempted; `tier` is the crosswalk's own
    verdict (`measured` / `topic-proxy`) on whether this KC is separable from
    its topic at all. Callers that display a number must show both.
    """
    params = bkt_mastery.params_for_level(getattr(user_state, "self_reported_level", None))
    row = _crosswalk().get(kc)
    if not row:
        return params.p_init, 0.0, "unmapped"

    atoms = row["atoms"]
    total_w = sum(float(a.get("w") or 0.0) for a in atoms)
    if total_w <= 0:
        return params.p_init, 0.0, row.get("tier") or "topic-proxy"

    acc = 0.0
    covered = 0.0
    for a in atoms:
        w = float(a.get("w") or 0.0)
        if w <= 0:
            continue
        atom_id = a.get("a")
        acc += w * bkt_mastery.current_mastery(
            user_state.atom_mastery, user_state.atom_last_ts, atom_id, params=params
        )
        if atom_id in (user_state.atom_mastery or {}):
            covered += w
    return acc / total_w, covered / total_w, row.get("tier") or "topic-proxy"


def kc_evidence_exhausted(user_state, kc: str) -> bool:
    """The learner has given every piece of evidence this concept can ask for.

    A KC's question pool is finite, and some are tiny — `numpy.sorting` owns one
    drill and the lattice root `numpy.ndarray-model` owns two. The atom-BKT bar
    (LEARNED_THRESHOLD, from a self-reported beginner's prior) wants roughly
    seven mastery-moving correct answers before it clears. A two-question
    concept therefore cannot reach it without re-serving the same two drills six
    times over, and until it clears, every concept downstream of it stays
    locked: at cold start the frontier is exactly one KC wide, so the whole
    448-question course sat behind two problems shown on a loop.

    Asking for more evidence than the bank can supply is not rigour, it is a
    dead end. So a concept also counts as learned when BOTH hold:

      * EVERY question the concept owns has been served — the pool has nothing
        new left to show them; and
      * the ladder has them on `solo`, its top rung, which by PROMOTE_LO takes
        four consecutive correct answers with no scaffold.

    The first clause has to be distinct-question coverage and not an attempt
    COUNT. Counting attempts reads as a coverage test and is not one: reaching
    `solo` already costs four attempts, so on the small pools this exists to
    rescue — `numpy.sorting` owns one drill, `numpy.linalg-basics` two — a
    count-based clause is satisfied before it is ever consulted and constrains
    nothing at all.

    The second clause is the performance bar, measured on the concept's own
    attempts. A small pool therefore buys a faster unlock but never a free one:
    missing once drops the rung, and the credit goes with it.
    """
    pool = set(questions_for_kc(kc))
    if not pool:
        return False
    if not pool <= _served_question_ids(user_state):
        return False
    if kc_stage(user_state, kc) != "solo":
        return False
    # Third clause (2026-08-30): the top rung's entry drill opens behind a
    # worked example (example_schedule.SCHEDULE["solo"]), and an answer read
    # off an example is not the test. The last UNAIDED_TO_FINISH answers at
    # `solo` must have been made without one — "you should really be tested
    # before you move on".
    return example_schedule.unaided_finish(
        ladder_view(user_state, kc).get("attempts") or [], "solo"
    )


def _served_question_ids(user_state) -> set:
    """Every question this learner has been handed, across all subtopics.

    A KC's questions can sit in more than one subtopic, so the union is the
    only honest answer to "has the bank shown them everything it has".
    """
    out = set()
    for sub_state in (getattr(user_state, "subtopic_states", None) or {}).values():
        out.update(getattr(sub_state, "served_question_ids", None) or ())
    return out


def kc_is_learned(user_state, kc: str) -> bool:
    if kc_mastery(user_state, kc)[0] >= LEARNED_THRESHOLD:
        return True
    return kc_evidence_exhausted(user_state, kc)


def kc_is_unlocked(user_state, kc: str) -> bool:
    """A KC is unlocked when every prerequisite is learned. Roots (no prereqs)
    are unlocked from the first session — they are the cold-start entry points,
    and there is always at least one or the course would be unenterable."""
    node = _registry().get(kc)
    if node is None:
        # A KC nothing in the registry knows about cannot be gated on. Serving
        # it is the lesser evil versus locking content out of reach entirely.
        return True
    return all(kc_is_learned(user_state, p) for p in node["prereqs"])


def question_kcs(qid: int) -> List[str]:
    row = _qmatrix().get(int(qid))
    return list(row["target_kcs"]) if row else []


def question_kc_gate(user_state, qid: int) -> bool:
    """Lattice gate for one question: servable iff every KC it targets has its
    prerequisites learned. Untagged questions pass — 75 of the 448 bank
    questions carry no q-matrix row, and locking them out entirely would remove
    content rather than order it, which ch. 32 is explicit about not doing."""
    kcs = question_kcs(qid)
    if not kcs:
        return True
    return all(kc_is_unlocked(user_state, k) for k in kcs)


# --- frontier and ordering ----------------------------------------------------


def frontier(user_state, require_questions: bool = True) -> List[str]:
    """The learner's knowledge frontier, in serving order.

    Membership: unlocked (all prerequisites learned) and not yet learned itself.
    Order: coreness first (descendant count — ch. 32's "appear more frequently
    as prerequisites"), then shallower prerequisite depth, then id for a stable
    tie-break so two runs of the same state never disagree.

    `require_questions` drops frontier KCs the bank cannot actually practise.
    They stay in the report — a frontier node with no questions is a content
    gap worth seeing — but the selector must not pick one and stall.
    """
    reg = _registry()
    if not reg:
        return []
    descendants, depth = _closure()
    by_kc = _questions_by_kc()

    out: List[str] = []
    for kc in reg:
        if kc_is_learned(user_state, kc):
            continue
        if not kc_is_unlocked(user_state, kc):
            continue
        if require_questions and not by_kc.get(kc):
            continue
        out.append(kc)
    out.sort(key=lambda k: (-descendants.get(k, 0), depth.get(k, 0), k))
    return out


def select_next_kc(user_state, eligible=None) -> Optional[str]:
    """The single KC the tutor intends to serve next — the one answer the
    graph's highlight and the practice queue must both use.

    `eligible(qid) -> bool` filters to questions that can actually be served
    right now (unserved, in a weighted subtopic). Without it this returns the
    frontier head, which is the *theoretical* next concept and can differ from
    what the queue can deliver: once the head's questions are all served but its
    mastery has not yet cleared the threshold, the queue moves to the next
    frontier KC while an eligibility-blind answer keeps naming the head. That
    divergence is what makes a highlight a lie, so callers that are about to
    serve a question must pass `eligible`.
    """
    for kc in frontier(user_state):
        qs = questions_for_kc(kc)
        if eligible is None or any(eligible(q) for q in qs):
            return kc
    return None


def questions_for_kc(kc: str) -> Sequence[int]:
    return _questions_by_kc().get(kc, ())


# Scaffolding rungs, in the order a learner should meet them. The KP frontmatter
# already declares which drills are faded (fill in a blank), guided, and
# independent (write it unaided); `build_qmatrix.py` records that choice as the
# tag's `source`. Ranking by it is what turns "the questions on this KC" into the
# expertise-reversal ladder the lesson author actually wrote.
#
# `leftover-assignment` is not a rung: those tags were hand-assigned to drills no
# KP references, so nothing has been faded for them. They sort last, after every
# authored rung.
_LADDER_RANK = {
    "kp-faded": 0,
    "kp-guided": 1,
    "kp-independent": 2,
    # Whole-KP problems (`integrated:` in the KP frontmatter). Its own rank
    # rather than a flag on `kp-independent`, because the top rung has to be
    # SELECTABLE — `lessons.is_integrated` could only ever relabel a question
    # the queue had already chosen for some other reason.
    "kp-integrated": 3,
}
_LADDER_UNRANKED = 4
# Public because `prioritization` has to be able to say "this question carries
# no rung at all" when it looks for spare work: anything with a rank belongs to
# some rung, and serving a rung the learner has not earned is exactly the
# promotion-on-exhaustion the ladder rewrite removes.
LADDER_UNRANKED = _LADDER_UNRANKED


def ladder_rank(qid: int) -> int:
    row = _qmatrix().get(int(qid)) or {}
    return _LADDER_RANK.get(row.get("source"), _LADDER_UNRANKED)


# ---------------------------------------------------------------------------
# Expertise-reversal ladder
#
# A concept is met three times with decreasing support:
#
#   worked      — a solved example, read not answered. Not graded.
#   faded       — the same shape of problem with the key step blanked out.
#   independent — write it unaided.
#
# Which rung a learner is on is decided per KC from that KC's OWN graded
# attempts, as an interval rather than a point estimate. Promotion requires the
# Wilson LOWER bound to clear PROMOTE_LO: being probably-fine is not enough to
# remove support, you have to be confidently fine. Demotion uses the UPPER
# bound, so support returns only when the learner is confidently struggling
# rather than on one unlucky answer — except for the immediate rule below,
# which is what a learner actually expects: get it wrong, see an example again.
#
# Why an interval and not a mastery cutoff. Delta-Learning's courses ladder
# (lib/courses-scaffold.ts) originally staged on a skill estimate and had to
# abandon it: faded attempts are recorded at a fixed low difficulty, so the
# skill estimate saturated below the promotion cutoff and learners were trapped
# on scaffolded cards forever. Scoring the ladder on the KC's own attempt
# record — not a global skill number the ladder itself depresses — is what
# avoids that trap here. The bounds below are calibrated so three consecutive
# correct answers promote (Wilson lower at 3/3 = 0.438), matching that project's
# 2-to-4 pacing, and four consecutive wrong answers drop the learner all the way
# back to full support (Wilson upper at 0/4 = 0.49).
#
# `worked` is entered once and never re-entered — see `_stage_from`. It is the
# teaching page rather than a drill, so demotion floors at `faded`, the lowest
# rung that is still a problem the learner answers.
LADDER_STAGES = ("worked", "faded", "partial", "solo")

# Wilson LOWER bound needed to climb off each rung. Calibrated against the
# bound at k/k so the pacing is legible in answers, not in probabilities:
#   faded   -> partial  at 0.34 = two consecutive correct
#   partial -> solo     at 0.51 = four consecutive correct
# which is Delta-Learning's PROMOTE_TO_PARTIAL=2 / PROMOTE_TO_SOLO=4 pacing,
# arrived at there by trial on real learners. Using the lower bound rather than
# the point estimate means a lucky streak of one does not strip support.
PROMOTE_LO = {"faded": 0.34, "partial": 0.51}
DEMOTE_HI = 0.50  # Wilson UPPER; 0/4 = 0.49, so four straight wrong restore support
_LADDER_WINDOW = 20  # recent attempts the estimate rests on
# Consecutive correct answers that promote a rung on their own, regardless of
# what the window average says — see `_streak_stage`. Three, to match the
# pacing the Wilson bounds were calibrated for (lower at 3/3 = 0.438) on a
# record that has no history to outvote it.
_PROMOTE_STREAK = 3


def _wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval. Wilson rather than the normal approximation
    because n here is tiny and p sits near the edges, where the normal interval
    runs outside [0, 1] and reports far more confidence than the data supports.
    """
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def ladder_row(user_state, kc: str) -> dict:
    store = getattr(user_state, "kc_ladder", None)
    if store is None:
        store = {}
        try:
            user_state.kc_ladder = store
        except Exception:
            pass
    row = store.get(kc)
    if not isinstance(row, dict):
        row = {"worked_seen": 0, "attempts": []}
        store[kc] = row
    row.setdefault("worked_seen", 0)
    row.setdefault("attempts", [])
    return row


def ladder_view(user_state, kc: str) -> dict:
    """A KC's ladder row for READING, without creating one that does not exist.

    `ladder_row` is the write path: it installs a fresh row in `kc_ladder` as a
    side effect, which is right when an attempt is about to be appended and
    wrong everywhere else. The lattice endpoint asks for every KC's stage and
    estimate on each load, so routing those reads through the write path would
    stamp all 63 concepts into every learner's stored state — sixty-three empty
    rows recording that nothing has happened, written on a GET.

    Returns a detached default for an absent row, so a caller that mutates the
    result cannot silently half-create state either.
    """
    store = getattr(user_state, "kc_ladder", None)
    row = store.get(kc) if isinstance(store, dict) else None
    if not isinstance(row, dict):
        return {"worked_seen": 0, "attempts": []}
    return row


def kc_estimate(user_state, kc: str) -> dict:
    """The learner's level on ONE concept, as an interval over its own attempts."""
    row = ladder_view(user_state, kc)
    attempts = row.get("attempts") or []
    recent = attempts[-_LADDER_WINDOW:]
    n = len(recent)
    k = sum(1 for a in recent if a.get("correct"))
    lo, hi = _wilson(k, n)
    # Trailing correct answers. This is the OTHER route out of a rung — see
    # `_streak_stage`: a run of `_PROMOTE_STREAK` promotes on its own, whatever
    # the window says, because a window with a long tail of old misses in it
    # never catches up with a learner who has plainly got it. Reported so the
    # UI can draw the route the learner is actually on; a run is the half of
    # promotion that moves with the answer in front of them.
    run = []
    for a in reversed(attempts):
        if not a.get("correct"):
            break
        run.append(a)
    est = {
        "n": n,
        "correct": k,
        "p": (k / n) if n else None,
        "ci": [round(lo, 4), round(hi, 4)],
        "streak": len(run),
        "streak_needed": _PROMOTE_STREAK,
        "worked_seen": int(row.get("worked_seen") or 0),
        # When this concept was last answered, from its own record. The graph's
        # "last practiced" line reads BKT atom timestamps, which exist for the
        # 20 crosswalked KCs and nowhere else — so a concept with nine attempts
        # behind it still displayed "never". This is the timestamp that always
        # exists when the record does.
        "last_ts": (recent[-1].get("ts") if n else None),
    }
    # Scoped to the rung the learner is standing on, which is computed from
    # everything above it. `_stage_from` does not read `streak`, so this is a
    # display correction and not a second promotion rule.
    stage = _stage_from(est, row)
    # The rung this estimate DESCRIBES, sent with it. A client can hold an
    # estimate that is newer than the rung on its screen — the practice strip
    # deliberately leaves the rung alone until the next question, because the
    # problem underneath it still came from the old one — and without this it
    # cannot tell "you have not left this rung yet" from "you left it with the
    # answer you just gave", which are opposite things to draw.
    est["stage"] = stage
    est["streak"] = _streak_toward(run, stage)
    return est


def _streak_toward(run: List[dict], stage: str) -> int:
    """Trailing correct answers that count toward LEAVING `stage`.

    `_streak_stage` promotes to one rung above the LOWEST rung in the run, so a
    run that already spans the promotion it bought is aiming at the rung the
    learner is already on: the streak route is spent until a miss resets the
    run. Reporting the raw run length there draws the newly reached rung
    arriving already full — three answers made at `faded` are not progress out
    of `partial`, and the strip would promise a promotion that cannot come.
    """
    ranks = [
        LADDER_STAGES.index(a["stage"])
        for a in run
        if a.get("stage") in LADDER_STAGES
    ]
    cur = LADDER_STAGES.index(stage) if stage in LADDER_STAGES else 0
    # The LENGTH counts every answer in the run and the RUNG is read only from
    # the ones filed at a ladder stage — the same split `_streak_stage` makes,
    # deliberately. `record_kc_outcome` files an off-ladder attempt (a placement
    # probe) as `independent`, and a run carrying one promotes on three exactly
    # as it does here. Counting the two differently is how a bar comes to fill
    # against a promotion that never arrives.
    return len(run) if ranks and min(ranks) >= cur else 0


def _step_down(stage: str, floor: str) -> str:
    """One rung down from `stage`, but never below `floor`.

    `floor` is required rather than defaulting to the bottom of the ladder:
    the bottom rung is the lesson page, and a caller that lands a learner
    there by accident re-teaches a concept they have already read.
    """
    i = LADDER_STAGES.index(stage) if stage in LADDER_STAGES else 1
    return LADDER_STAGES[max(LADDER_STAGES.index(floor), i - 1)]


def _step_up(stage: str, ceiling: str = "solo") -> str:
    """One rung up from `stage`, but never above `ceiling`."""
    i = LADDER_STAGES.index(stage) if stage in LADDER_STAGES else 1
    return LADDER_STAGES[min(LADDER_STAGES.index(ceiling), i + 1)]


def _streak_stage(attempts: List[dict]) -> Optional[str]:
    """The rung earned by an unbroken run of correct answers, or None.

    WHY THE WILSON BOUND IS NOT ENOUGH ON ITS OWN.
    The interval is computed over the last `_LADDER_WINDOW` attempts, and the
    calibration note above the thresholds says three consecutive correct answers
    should promote — which is true of an EMPTY record (Wilson lower at 3/3 =
    0.438) and false of every record with history in it. A learner who met a
    concept when its drills were badly written can bank a dozen misses, fix
    nothing about their own understanding except everything, and then answer
    five in a row without moving: 7/20 has a lower bound of 0.18 against a
    `faded` bar of 0.34, so the rung does not move for another dozen questions.
    That is a real record from this app, not a hypothetical.

    The asymmetry is the giveaway. Demotion already has an immediate rule — miss
    one, drop a rung, see the support again — so the ladder listens to a single
    recent answer on the way down and to a twenty-question average on the way
    up. A learner who has clearly got it is then told, question after question,
    that they have not.

    So a run of correct answers promotes one rung on its own. It reads the rung
    the run was actually MADE at (the lowest in the run, so a streak spanning a
    promotion cannot skip the rung it just reached) and steps up once from
    there. One rung, never two: the evidence is that this rung is done, not that
    the one above it is.

    Bounded on the other side by the demotion rule, which is unchanged and still
    immediate — a streak buys one rung and one miss gives it straight back.
    """
    run = []
    for attempt in reversed(attempts):
        if not attempt.get("correct"):
            break
        run.append(attempt)
    if len(run) < _PROMOTE_STREAK:
        return None
    ranks = [
        LADDER_STAGES.index(a.get("stage"))
        for a in run
        if a.get("stage") in LADDER_STAGES
    ]
    if not ranks:
        return None
    return _step_up(LADDER_STAGES[min(ranks)])


def _stage_from(est: dict, row: dict) -> str:
    """The rung, given a KC's estimate and its ladder row.

    Split out from `kc_stage` so a caller holding both can reuse them. The
    lattice report wants the stage AND the estimate for all 63 concepts, and
    going through `kc_stage` there would recompute the estimate — the window
    slice, the count and the Wilson interval — a second time for every row.

    ONE-WAY DOOR AT `worked`. This rung is not a drill: it is the teaching
    page, the one `LessonGate` takes over the screen to show. Every path below
    the cold-start check therefore floors at `faded`, so a learner who has
    already been taught a concept is never handed its lesson again by the
    scheduler. Support still comes back on a miss — `faded` keeps the worked
    example on screen beside the problem (LadderUI.SUPPORTED_STAGES) — but the
    thing that comes back is the example, not the explanation.

    Why not re-teach on a bad streak. The learner has read this page; replaying
    it is the system asserting they did not, which is both wrong (a miss says
    they cannot yet APPLY the idea, not that they never met it) and unskippable
    — the demotion re-fires from `attempts[-1]` on every subsequent question,
    so one miss put the lesson in front of them again and again until they
    happened to answer correctly. Re-reading a lesson stays available, but as
    something the learner chooses: the concept graph's node opens it, and so
    does `?lesson=<kc>`.
    """
    # Cold start: the example comes first, always. A concept nobody has been
    # shown cannot be assessed, and guessing at it is not assessment. This is
    # the only branch that may return `worked`.
    if est["worked_seen"] == 0:
        return "worked"

    attempts = row.get("attempts") or []
    if attempts and not attempts[-1].get("correct"):
        # The rule a learner actually expects: miss one, drop back a rung and
        # see the support again. Stepping down from the stage the MISSED
        # attempt was made at, not from today's computed stage, so a wrong
        # answer on an independent problem lands on faded rather than skipping
        # straight back past the scaffolded rungs.
        return _step_down(attempts[-1].get("stage") or "faded", floor="faded")

    lo, hi = est["ci"]
    # A run of correct answers earns a rung on its own — the window average is
    # slow to forget, and on a concept with a long tail of old misses it never
    # catches up with a learner who has plainly got it. Read before the
    # struggling branch: `hi < DEMOTE_HI` is a statement about the same stale
    # window, and it must not hold a learner down who is currently on a run.
    streak = _streak_stage(attempts)
    if est["n"] and hi < DEMOTE_HI and not streak:
        # Confidently struggling: all the support there is, which is the fully
        # visible worked example at `faded`.
        return "faded"
    # Climb one rung at a time: clearing the solo bar also clears the partial
    # bar, so test from the top down and take the highest rung earned.
    if lo >= PROMOTE_LO["partial"]:
        earned = "solo"
    elif lo >= PROMOTE_LO["faded"]:
        earned = "partial"
    else:
        earned = "faded"
    # Whichever route got further. They measure different things — one that the
    # concept is reliably right over a window, one that it is right NOW — and
    # taking the lower would let a poisoned window cancel a run it cannot see.
    if streak and LADDER_STAGES.index(streak) > LADDER_STAGES.index(earned):
        return streak
    return earned


def kc_stage(user_state, kc: str) -> str:
    """Which rung to serve for this concept right now."""
    return _stage_from(kc_estimate(user_state, kc), ladder_view(user_state, kc))


def note_worked_seen(user_state, kc: str) -> None:
    ladder_row(user_state, kc)["worked_seen"] += 1


def record_kc_outcome(
    user_state, qid: int, correct: bool, stage: str = "independent", example: bool = False
) -> List[str]:
    """Log a graded attempt against every KC the question targets.

    `example` is whether the drill was served behind a worked-example popup
    (example_schedule.plan). Stored with the attempt because the schedule reads
    it back — a miss made behind an example does not buy a second example —
    and because kc_evidence_exhausted requires the last top-rung answers to
    have been made without one.

    Returns the KCs touched, so a caller can report what moved. Placement
    probes must NOT come through here: the diagnostic measures prior knowledge,
    and feeding ungated probes into the ladder would demote a learner to worked
    examples for concepts the probe was never trying to teach.
    """
    touched = question_kcs(qid)
    for kc in touched:
        row = ladder_row(user_state, kc)
        row["attempts"].append({
            "correct": bool(correct),
            "stage": stage if stage in LADDER_STAGES else "independent",
            "ts": datetime.now(timezone.utc).isoformat(),
            # Which question this row is about. The ladder itself never needs
            # it — it counts outcomes at rungs — but this row is the ONLY
            # durable record of the rung a question was actually served at, and
            # `engine_bridge.served_stage` reads it. Without the id, "the newest
            # row" is the best a reader can do, and on a route that does not
            # append here (a placement probe) that silently means "some earlier
            # question's rung". Rows written before this field simply do not
            # match, which is the safe direction: the engine declines the
            # attempt rather than scoring it at a rung it is guessing.
            "question_id": int(qid),
            "example": bool(example),
        })
        del row["attempts"][:-_LADDER_WINDOW]
    return touched


# Which authored rung serves which ladder stage, IN PREFERENCE ORDER. "worked"
# is not a drill at all — it is the lesson page the frontend renders — so it has
# no entry and never selects a question.
#
# The four rungs the learner is shown, and what each one puts on screen
# (2026-08-28, from Seth's own description of the loop he wants):
#
#   worked  / Lesson      the KP's pages, examples and all. Not graded.
#   faded   / Faded       the problem, and a starter with the new syntax blanked
#                         out. NO example beside it — the learner met the
#                         examples in the lesson they just read, and one sitting
#                         next to the blanks spells the answer (which is exactly
#                         the defect reported against q484).
#   partial / Solo        write the whole function unaided. An example appears
#                         above SOME of these — the ones whose KP authored a
#                         ```python worked``` fence for them — and the queue
#                         serves those first, so examples thin out across the
#                         rung instead of stopping on a cutoff.
#   solo    / Integrated  whole-KP problems, every concept of the KP at once,
#                         and never an example.
#
# The STORED stage strings are unchanged on purpose. Every attempt a learner has
# ever made is filed under one of these four names and the promotion arithmetic
# reads them back; renaming them would either rewrite that history or silently
# reset it. What changed is what each rung serves and what it is called on
# screen (practice/stage-ladder.js).
#
# `guided` sits with `faded`: a guided item carries hints and a derived
# blank-starter, which is faded-rung support by another name. `partial` falls
# back to unranked leftovers, and `solo` falls back to `kp-independent` and then
# leftovers, so a KP that authored no `## Integrated practice` still has a top
# rung to serve.
_STAGE_TO_RANKS = {
    "faded": (0, 1),
    "partial": (2, _LADDER_UNRANKED),
    "solo": (3, 2, _LADDER_UNRANKED),
}


# The rungs that put support in front of the learner: an authored faded drill
# carries the `_____` blanks, a guided one carries hints. Independent,
# integrated and unranked carry neither.
_SUPPORTED_RANKS = frozenset({0, 1})


def stage_requires_support(stage: str) -> bool:
    """Does this stage promise the learner something to work from?

    Only `faded` does now: the blanks, or a guided item's hints. `partial` used
    to count because an example sat above the problem, but that example is
    authored per ITEM rather than per rung — most solo drills have none — so it
    was never a promise the rung could keep.

    Kept because `lessons.rung_support` still reports, per card, whether the
    promise this rung makes is on the page; the strip draws it.
    """
    ranks = _STAGE_TO_RANKS.get(stage)
    if not ranks:
        return False
    return set(ranks) <= _SUPPORTED_RANKS


def questions_at_stage(qids: Iterable[int], stage: str) -> List[int]:
    """The questions this rung serves, from the FIRST authored rank that has any.

    Preference order, not a set union. `solo` prefers a whole-KP `integrated`
    problem and settles for an `independent` one; `faded` prefers a hand-cut
    faded drill and settles for a guided one. Taking the union instead would let
    the difficulty picker hand a learner on the top rung a single-concept drill
    while a whole-KP problem sat unserved beside it.

    Returns [] when the stage has nothing — the caller reads that as "this rung
    is spent", which is now a reportable state rather than a licence to repeat.
    """
    ranks = _STAGE_TO_RANKS.get(stage)
    if not ranks:
        return []
    pool = list(qids)
    for rank in ranks:
        at_rank = [q for q in pool if ladder_rank(q) == rank]
        if at_rank:
            return at_rank
    return []


def with_example_first(qids: Iterable[int]) -> List[int]:
    """The rung's example-bearing drills, if it still has any unserved.

    THE FADE FROM `partial` TO `solo`, and it needs no schedule and no counter.
    A solo drill carries an example only when its KP authored one for it (six of
    nineteen on `numpy.ndarray-model`), and the queue never repeats a served
    question — so serving the example-bearing ones first means a learner meets
    an example, then a few unaided problems, then another example introducing
    the next move, and then none at all once they are spent. "It shows examples
    less and less until it doesn't show any examples at all."

    Returns the whole input when no example-bearing drill is left, which is the
    far end of that fade rather than a failure.
    """
    pool = list(qids)
    with_example = [q for q in pool if lessons.has_worked_example(q)]
    return with_example or pool


def lowest_rung(qids: Iterable[int]) -> List[int]:
    """The questions on the least-scaffolded rung still available.

    Called with the questions a learner can actually be served right now, so an
    exhausted faded rung falls through to guided and then independent instead of
    dead-ending. Returns [] for an empty input, which callers read as "this KC
    has nothing left" rather than as an ordering result.
    """
    pool = list(qids)
    if not pool:
        return []
    floor = min(ladder_rank(q) for q in pool)
    return [q for q in pool if ladder_rank(q) == floor]


def registry_node(kc: str) -> Optional[dict]:
    """The KC's registry entry (title, lesson, topic, prereqs), or None."""
    return _registry().get(kc)


def crosswalk_row(kc: str) -> Optional[dict]:
    """The KC's crosswalk entry — its weighted atoms and its tier — or None.

    A public reader for the same table `kc_mastery` consumes, so a caller that
    wants the atoms rather than the aggregate does not have to reach into
    `_crosswalk`. `engine_bridge` wants exactly that: the logistic engine's
    `encompassing` feature is a mean over the atoms, not the KC's own number.
    """
    return _crosswalk().get(kc)


def subtopics_for_kc(kc: str) -> List[str]:
    """Subtopic keys that carry this KC's questions, registry key first.

    The registry's `subtopic_key` is the authored intent; the q-matrix is what
    the bank actually does. They can disagree, so both are returned and the
    caller checks which one has servable questions.
    """
    out: List[str] = []
    node = _registry().get(kc)
    if node and node.get("subtopic_key"):
        out.append(node["subtopic_key"])
    return out


def kc_report(user_state, eligible=None) -> dict:
    """Full lattice state for the API — one row per KC, plus the selection the
    queue will actually make. This is what lets the knowledge graph draw the
    system's real state instead of a decorative model: every field the graph
    colours by is computed here, by the same code that gates practice.
    """
    reg = _registry()
    descendants, depth = _closure()
    by_kc = _questions_by_kc()
    order = {kc: i for i, kc in enumerate(frontier(user_state))}
    # `eligible` makes the reported next_kc the one the QUEUE will reach, not
    # the frontier head it would like to reach. Without it the graph can ring a
    # concept whose questions are all spent while practice serves the next one
    # along — a highlight that promises something the app then does not do.
    next_kc = select_next_kc(user_state, eligible=eligible)

    rows = {}
    for kc, node in reg.items():
        m, covered, tier = kc_mastery(user_state, kc)
        # Same predicate the practice gate uses, exhaustion credit included —
        # a node the queue treats as cleared must not draw as still-frontier.
        learned = kc_is_learned(user_state, kc)
        unlocked = kc_is_unlocked(user_state, kc)
        ladder_est = kc_estimate(user_state, kc)
        # The concept's OWN graded record, alongside the crosswalk mastery.
        #
        # These answer different questions and the graph has only ever had the
        # first. `mastery` is a BKT posterior over the ATOMS a KC's questions
        # exercise, and the crosswalk that joins concepts to atoms separates
        # only 20 of 63 — for the other 43 the number shown is the topic's,
        # which is why a learner with real drill history still saw bubbles that
        # claimed nothing had been measured about them.
        #
        # The ladder record has no such gap: `record_ladder_outcome` writes one
        # row per graded attempt against every KC the question tags, for all 63.
        # It is a smaller claim than a posterior — k correct out of n, on this
        # concept, recently — but it is this concept's, always, and it is the
        # same quantity the practice topbar draws and the rung gate promotes on.
        # Shipping it here is what lets the graph and the practice screen agree.
        rows[kc] = {
            "title": node["title"],
            "lesson": node["lesson"],
            "topic": node["topic"],
            "prereqs": node["prereqs"],
            "mastery": round(m, 4),
            "covered_w": round(covered, 3),
            "tier": tier,
            "evidenced": covered >= MIN_COVERED_W,
            "state": "learned" if learned else ("frontier" if unlocked else "locked"),
            "coreness": descendants.get(kc, 0),
            "depth": depth.get(kc, 0),
            "n_questions": len(by_kc.get(kc, ())),
            "frontier_rank": order.get(kc),
            "ladder_stage": _stage_from(ladder_est, ladder_view(user_state, kc)),
            "ladder_estimate": ladder_est,
        }

    return {
        "learned_threshold": LEARNED_THRESHOLD,
        "next_kc": next_kc,
        "frontier": frontier(user_state),
        "kcs": rows,
    }
