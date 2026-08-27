"""
Concept-graph structure feedback endpoint — the instructor review surface's
sibling to problem_feedback_router.py.

Lets an instructor (anyone with instructor feedback mode on — it is a workflow
choice, not a privilege) flag the STRUCTURE of the concept graph: an edge that
points the wrong way or should not exist, a node that is mislabeled or filed
under the wrong topic, an edge that is MISSING between two concepts, or a
concept that is missing altogether. This is
sequencing feedback (is the *graph* right?), NOT content feedback about any one
question (that is problem_feedback_router.py) and NOT the difficulty rating
that feeds the adaptive engine (feedback_router.py).

Deliberately the same shape as the problem-feedback log, for the same reason:
entries are appended to a SIBLING file `{user_id}.graph_feedback.json` in the
practice data dir. It never reads or writes UserPracticeState or the graph
itself — a submission is a claim for a human to review, not an edit. There is
no repair queue behind this one (an edge change reshapes the unlock lattice for
everyone; nothing automated should touch it): the log IS the deliverable, and
`grep`/a future review reader is the consumer.

Endpoints (mounted under /api/practice by the parent router):
  POST /graph-feedback -> append one entry
  GET  /graph-feedback -> list this user's entries (newest first)

`missing_node` is the one kind whose `source` is not an id that exists: the
instructor drew a concept the curriculum does not have yet, so `source` carries
the client-minted placeholder id it was drawn under ("new:<slug>"), `label`
carries what they named it, and `target` names the existing concept it was
drawn hanging off — which is the proposed placement, so a maintainer can accept
the concept and reject where it went. The connecting arrow is NOT a second
`missing_edge` row: an edge to a concept that does not exist yet is not a claim
anyone can act on separately.

Node/edge ids are the string ids from the graph named in `graph`. They are
NOT validated against the live graph on purpose: the frontend bakes a graph
snapshot, the backend's may differ mid-deploy, and a flag on a node that was
just renamed is exactly the feedback worth keeping.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.adaptive import DATA_DIR
from app.auth import get_current_user
from app.models import User

logger = logging.getLogger(__name__)

router = APIRouter()

GraphFeedbackKind = Literal["edge", "node", "missing_edge", "missing_node"]

# One flat tag vocabulary across kinds; the frontend offers the subset that
# makes sense for what was tapped. "proposed" is the missing_edge submission
# itself (the edge_type field carries whether it should be prereq or enc).
GraphFeedbackTag = Literal[
    "wrong_direction",
    "should_not_exist",
    "wrong_type",
    "mislabeled",
    "wrong_topic",
    "good",
    "proposed",
]


# WHICH graph an id names. Closed, not free text: the whole point of the field
# is to tell two id spaces apart, and a namespace that accepts a typo is a
# namespace that does not.
GraphNamespace = Literal["lesson-kc", "arena-atom"]


class GraphFeedbackRequest(BaseModel):
    kind: GraphFeedbackKind
    source: str = Field(min_length=1, max_length=200)
    # Required for kind=edge and kind=missing_edge; absent for kind=node.
    target: Optional[str] = Field(default=None, max_length=200)
    # For edges: which lane the flag is about / the proposed edge should be.
    edge_type: Optional[Literal["prereq", "enc"]] = None
    tag: GraphFeedbackTag
    note: str = Field(default="", max_length=5000)
    # WHICH graph the ids name. The instructor surface used to flag the ARENA
    # atom graph and now flags the lesson graph (kc_registry.json ids), and the
    # two id spaces do not overlap — a log that does not say which one it is
    # cannot be read a month later. Optional so an older client, and every
    # entry already on disk, stays valid.
    graph: Optional[GraphNamespace] = None
    # What a PROPOSED concept was named. Only `missing_node` fills it — every
    # other kind names things that already exist, and their label is whatever
    # the registry says today, which is the reading a maintainer should get
    # rather than a snapshot of what it said when the flag was filed.
    label: Optional[str] = Field(default=None, max_length=200)


class GraphFeedbackEntry(BaseModel):
    kind: GraphFeedbackKind
    source: str
    target: Optional[str] = None
    edge_type: Optional[str] = None
    tag: str
    note: str = ""
    graph: Optional[str] = None
    label: Optional[str] = None
    timestamp: str


class GraphFeedbackResponse(BaseModel):
    success: bool
    count: int


class GraphFeedbackListResponse(BaseModel):
    entries: List[GraphFeedbackEntry]


def _log_file(user_id: str):
    safe_id = user_id.replace("/", "_").replace("..", "_")
    return DATA_DIR / f"{safe_id}.graph_feedback.json"


def _read_entries(user_id: str) -> List[dict]:
    path = _log_file(user_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("entries") or [])
    except Exception as e:  # a corrupt log must not break feedback submission
        logger.error("Failed to read graph-feedback log for %s: %s", user_id, e)
        return []


def _write_entries(user_id: str, entries: List[dict]) -> None:
    path = _log_file(user_id)
    payload = {"user_id": user_id, "entries": entries}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@router.post("/graph-feedback", response_model=GraphFeedbackResponse)
def submit_graph_feedback(
    payload: GraphFeedbackRequest,
    user: User = Depends(get_current_user),
) -> GraphFeedbackResponse:
    """Append one graph-structure flag. Append-only; nothing else moves."""
    if payload.kind in ("edge", "missing_edge") and not (payload.target or "").strip():
        raise HTTPException(status_code=422, detail="target is required for edge feedback")
    if payload.kind in ("missing_edge", "missing_node") and payload.tag != "proposed":
        raise HTTPException(status_code=422, detail="a missing edge or node is submitted as tag=proposed")
    # A proposed concept's `source` is a placeholder id this request minted, so
    # the NAME is the only part of it anyone can act on. Without it the entry is
    # a row saying a concept is missing without saying which — unreadable a
    # month later, and indistinguishable from a client bug.
    if payload.kind == "missing_node":
        if not (payload.label or "").strip():
            raise HTTPException(status_code=422, detail="label is required for a proposed concept")
        # And where it goes. A concept with no placement is a wish, not a
        # proposal — a maintainer cannot accept or reject it, and the client
        # always has one (a proposed concept is drawn hanging off a bubble).
        if not (payload.target or "").strip():
            raise HTTPException(status_code=422, detail="target is required for a proposed concept")
    user_id = str(user.id)
    entries = _read_entries(user_id)
    entry = {
        "kind": payload.kind,
        "source": payload.source.strip(),
        "target": (payload.target or "").strip() or None,
        "edge_type": payload.edge_type,
        "tag": payload.tag,
        "note": payload.note.strip(),
        "graph": payload.graph or None,
        "label": (payload.label or "").strip() or None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    entries.append(entry)
    _write_entries(user_id, entries)
    logger.info(
        "graph_feedback user=%s kind=%s %s->%s tag=%s note=%r",
        user_id, payload.kind, entry["source"], entry["target"], payload.tag,
        entry["note"][:120],
    )
    return GraphFeedbackResponse(success=True, count=len(entries))


@router.get("/graph-feedback", response_model=GraphFeedbackListResponse)
def list_graph_feedback(
    user: User = Depends(get_current_user),
) -> GraphFeedbackListResponse:
    entries = _read_entries(str(user.id))
    entries.reverse()  # newest first
    return GraphFeedbackListResponse(
        entries=[GraphFeedbackEntry(**e) for e in entries]
    )
