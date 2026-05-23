"""
Concept-graph schema and loader for graph-backed curriculum sequencing.

This is the first implementation step for the graph-based tutor plan:
  - explicit concept schema
  - prerequisite edges
  - core vs supplemental tagging
  - mastery-gated advancement
  - lesson and problem-to-concept links

The initial implementation is JSON-backed on purpose. It is easy to inspect,
version, hand-seed, and revise before committing to a database migration.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Literal

from pydantic import BaseModel, Field

_THIS_DIR_ONLY = Path(__file__).resolve().parents[3] / "This-Directory-Only"
DEFAULT_GRAPH_PATH = (
    _THIS_DIR_ONLY
    / "backend"
    / "app"
    / "data"
    / "concept_graphs"
    / "arena_prereqs_einops_foundations.json"
)

NodeType = Literal["concept"]
ConceptKind = Literal["knowledge", "skill", "strategy"]
ConceptTier = Literal["core", "supplemental"]
LessonRole = Literal["introduce", "practice", "diagnostic", "review"]
ProblemLinkRole = Literal["primary", "supporting", "diagnostic", "transfer", "review"]


class ConceptNode(BaseModel):
    id: str
    title: str
    topic: str
    subtopic: str
    node_type: NodeType = "concept"
    kind: ConceptKind = "skill"
    tier: ConceptTier = "core"
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class PrerequisiteEdge(BaseModel):
    prerequisite_id: str
    dependent_id: str
    weight: float = Field(ge=0.0, le=1.0, default=1.0)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    is_hard_gate: bool = True
    rationale: str = ""


class LessonNode(BaseModel):
    id: str
    title: str
    summary: str
    topic: str
    subtopic: str
    tier: ConceptTier = "core"
    source_kind: str
    source_ref: str
    concept_ids: list[str] = Field(default_factory=list)
    lesson_role: LessonRole = "introduce"


class ProblemConceptLink(BaseModel):
    source_bank: str
    problem_ref: str
    concept_id: str
    role: ProblemLinkRole = "primary"
    weight: float = Field(ge=0.0, le=1.0, default=1.0)
    notes: str = ""


class MasteryGatePolicy(BaseModel):
    mastery_threshold: float = Field(ge=0.0, le=1.0, default=0.8)
    require_all_hard_prereqs: bool = True
    supplemental_concepts_block_progress: bool = False
    notes: str = ""


class CurriculumGraph(BaseModel):
    curriculum_id: str
    title: str
    version: str
    description: str
    concepts: list[ConceptNode] = Field(default_factory=list)
    prerequisite_edges: list[PrerequisiteEdge] = Field(default_factory=list)
    lessons: list[LessonNode] = Field(default_factory=list)
    problem_links: list[ProblemConceptLink] = Field(default_factory=list)
    mastery_gate: MasteryGatePolicy = Field(default_factory=MasteryGatePolicy)


def _validate_graph(graph: CurriculumGraph) -> CurriculumGraph:
    concept_ids = {concept.id for concept in graph.concepts}
    if len(concept_ids) != len(graph.concepts):
        raise ValueError("Duplicate concept id detected in curriculum graph")

    lesson_ids = {lesson.id for lesson in graph.lessons}
    if len(lesson_ids) != len(graph.lessons):
        raise ValueError("Duplicate lesson id detected in curriculum graph")

    for edge in graph.prerequisite_edges:
        if edge.prerequisite_id not in concept_ids:
            raise ValueError(f"Unknown prerequisite concept id: {edge.prerequisite_id}")
        if edge.dependent_id not in concept_ids:
            raise ValueError(f"Unknown dependent concept id: {edge.dependent_id}")

    for lesson in graph.lessons:
        missing = [concept_id for concept_id in lesson.concept_ids if concept_id not in concept_ids]
        if missing:
            raise ValueError(f"Lesson {lesson.id} references unknown concepts: {missing}")

    for link in graph.problem_links:
        if link.concept_id not in concept_ids:
            raise ValueError(
                f"Problem link {link.source_bank}:{link.problem_ref} references unknown concept {link.concept_id}"
            )

    return graph


@lru_cache(maxsize=8)
def load_curriculum_graph(path: str | Path | None = None) -> CurriculumGraph:
    graph_path = Path(path).resolve() if path else DEFAULT_GRAPH_PATH
    raw = json.loads(graph_path.read_text(encoding="utf-8"))
    return _validate_graph(CurriculumGraph.model_validate(raw))


def concept_index(graph: CurriculumGraph) -> Dict[str, ConceptNode]:
    return {concept.id: concept for concept in graph.concepts}


def prerequisite_edges_for(graph: CurriculumGraph, concept_id: str) -> list[PrerequisiteEdge]:
    return [edge for edge in graph.prerequisite_edges if edge.dependent_id == concept_id]


def lesson_index(graph: CurriculumGraph) -> Dict[str, LessonNode]:
    return {lesson.id: lesson for lesson in graph.lessons}


def problem_links_for_concept(graph: CurriculumGraph, concept_id: str) -> list[ProblemConceptLink]:
    return [link for link in graph.problem_links if link.concept_id == concept_id]


def mastery_gated_ready_concepts(
    graph: CurriculumGraph,
    mastery_by_concept: Dict[str, float],
    *,
    include_mastered: bool = False,
) -> list[ConceptNode]:
    ready: list[ConceptNode] = []
    for concept in graph.concepts:
        if not include_mastered and mastery_by_concept.get(concept.id, 0.0) >= graph.mastery_gate.mastery_threshold:
            continue
        if concept_is_unlocked(graph, concept.id, mastery_by_concept):
            ready.append(concept)
    ready.sort(key=lambda item: (item.tier != "core", item.topic, item.subtopic, item.title))
    return ready


def concept_is_unlocked(
    graph: CurriculumGraph,
    concept_id: str,
    mastery_by_concept: Dict[str, float],
) -> bool:
    concept = concept_index(graph).get(concept_id)
    if concept is None:
        raise KeyError(f"Unknown concept id: {concept_id}")

    prereqs = prerequisite_edges_for(graph, concept_id)
    if not prereqs:
        return True

    threshold = graph.mastery_gate.mastery_threshold
    hard_gate_edges = [edge for edge in prereqs if edge.is_hard_gate]
    soft_gate_edges = [edge for edge in prereqs if not edge.is_hard_gate]

    if graph.mastery_gate.require_all_hard_prereqs:
        hard_ready = all(mastery_by_concept.get(edge.prerequisite_id, 0.0) >= threshold for edge in hard_gate_edges)
    else:
        hard_ready = any(mastery_by_concept.get(edge.prerequisite_id, 0.0) >= threshold for edge in hard_gate_edges)

    if not hard_ready:
        return False

    if concept.tier == "supplemental" and not graph.mastery_gate.supplemental_concepts_block_progress:
        return True

    if not soft_gate_edges:
        return True

    # Soft gates are advisory: require at least one of them for core concepts.
    return any(mastery_by_concept.get(edge.prerequisite_id, 0.0) >= threshold for edge in soft_gate_edges)


def mastery_gate_snapshot(
    graph: CurriculumGraph,
    mastery_by_concept: Dict[str, float],
) -> list[dict]:
    """Explain why each concept is or is not unlocked."""
    concepts = concept_index(graph)
    snapshot: list[dict] = []
    for concept in graph.concepts:
        prereqs = prerequisite_edges_for(graph, concept.id)
        snapshot.append(
            {
                "concept_id": concept.id,
                "title": concept.title,
                "tier": concept.tier,
                "mastery": mastery_by_concept.get(concept.id, 0.0),
                "unlocked": concept_is_unlocked(graph, concept.id, mastery_by_concept),
                "prerequisites": [
                    {
                        "concept_id": edge.prerequisite_id,
                        "title": concepts[edge.prerequisite_id].title,
                        "mastery": mastery_by_concept.get(edge.prerequisite_id, 0.0),
                        "threshold": graph.mastery_gate.mastery_threshold,
                        "is_hard_gate": edge.is_hard_gate,
                        "weight": edge.weight,
                    }
                    for edge in prereqs
                ],
            }
        )
    return snapshot


def iter_seed_graph_paths() -> Iterable[Path]:
    graph_dir = DEFAULT_GRAPH_PATH.parent
    if not graph_dir.exists():
        return []
    return sorted(graph_dir.glob("*.json"))
