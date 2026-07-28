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
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

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
    """Question id -> {target_kcs, supporting_kcs}."""
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


def kc_is_learned(user_state, kc: str) -> bool:
    return kc_mastery(user_state, kc)[0] >= LEARNED_THRESHOLD


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


def select_next_kc(user_state) -> Optional[str]:
    """The single KC the tutor intends to serve next. This is the one function
    the graph's highlight and the practice queue must agree on; anything that
    computes its own answer will drift from what the learner is actually shown."""
    f = frontier(user_state)
    return f[0] if f else None


def questions_for_kc(kc: str) -> Sequence[int]:
    return _questions_by_kc().get(kc, ())


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


def kc_report(user_state) -> dict:
    """Full lattice state for the API — one row per KC, plus the selection the
    queue will actually make. This is what lets the knowledge graph draw the
    system's real state instead of a decorative model: every field the graph
    colours by is computed here, by the same code that gates practice.
    """
    reg = _registry()
    descendants, depth = _closure()
    by_kc = _questions_by_kc()
    order = {kc: i for i, kc in enumerate(frontier(user_state))}

    rows = {}
    for kc, node in reg.items():
        m, covered, tier = kc_mastery(user_state, kc)
        learned = m >= LEARNED_THRESHOLD
        unlocked = kc_is_unlocked(user_state, kc)
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
        }

    return {
        "learned_threshold": LEARNED_THRESHOLD,
        "next_kc": select_next_kc(user_state),
        "frontier": frontier(user_state),
        "kcs": rows,
    }
