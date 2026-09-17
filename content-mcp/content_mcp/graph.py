"""The concept graph: `lessons/kc_registry.json`.

Two things live in that file and both matter. The `prereqs` lists are the
lattice — what a learner must know before a concept. The ORDER of the `kcs`
array is the teaching sequence, and the repo asserts at load that the sequence
is a linear extension of the lattice: every prerequisite appears before the
concept that needs it. A write that breaks either is refused here rather than
discovered by a failing validator three steps later.
"""

from __future__ import annotations

import json

from . import lessons, paths


def load() -> dict:
    return json.loads(paths.KC_REGISTRY.read_text())


def _save(registry: dict) -> None:
    indent = paths.json_indent_of(paths.KC_REGISTRY)
    paths.KC_REGISTRY.write_text(json.dumps(registry, indent=indent) + "\n")


def _by_id(registry: dict) -> dict[str, dict]:
    return {kc["id"]: kc for kc in registry["kcs"]}


def check(registry: dict) -> list[str]:
    """Every structural rule the registry must satisfy. Empty list = good."""
    problems: list[str] = []
    known = _by_id(registry)
    lesson_ids = {lesson["id"] for lesson in registry["lessons"]}

    seen: set[str] = set()
    for kc in registry["kcs"]:
        if kc["id"] in seen:
            problems.append(f"duplicate KC id: {kc['id']}")
        seen.add(kc["id"])
        if kc.get("lesson") not in lesson_ids:
            problems.append(f"{kc['id']} names unknown lesson '{kc.get('lesson')}'")
        for prereq in kc.get("prereqs", []):
            if prereq not in known:
                problems.append(f"{kc['id']} lists unknown prereq '{prereq}'")
            elif prereq not in seen:
                problems.append(
                    f"order: {kc['id']} comes before its prereq '{prereq}' — "
                    "registry order must be a linear extension of the lattice"
                )
        if kc["id"] in kc.get("prereqs", []):
            problems.append(f"{kc['id']} is its own prereq")

    # A cycle cannot survive the order check above, but check explicitly so the
    # message says "cycle" when someone reorders and re-runs.
    colour: dict[str, int] = {}

    def visit(node: str, trail: list[str]) -> None:
        if colour.get(node) == 2:
            return
        if colour.get(node) == 1:
            problems.append("cycle: " + " -> ".join(trail + [node]))
            return
        colour[node] = 1
        for prereq in known.get(node, {}).get("prereqs", []):
            if prereq in known:
                visit(prereq, trail + [node])
        colour[node] = 2

    for kc in registry["kcs"]:
        visit(kc["id"], [])
    return problems


def _drill_counts() -> dict[str, int]:
    if not paths.QMATRIX.exists():
        return {}
    tags = json.loads(paths.QMATRIX.read_text())
    counts: dict[str, int] = {}
    records = tags.get("questions", tags) if isinstance(tags, dict) else tags
    if isinstance(records, dict):
        records = list(records.values())
    for record in records or []:
        for kc in (record or {}).get("target_kcs", []) if isinstance(record, dict) else []:
            counts[kc] = counts.get(kc, 0) + 1
    return counts


def list_graph(lesson: str | None = None) -> dict:
    registry = load()
    pages = lessons.index()
    counts = _drill_counts()
    nodes = []
    for position, kc in enumerate(registry["kcs"]):
        if lesson and kc.get("lesson") != lesson:
            continue
        nodes.append(
            {
                "id": kc["id"],
                "position": position,
                "lesson": kc.get("lesson"),
                "title": kc.get("title"),
                "prereqs": kc.get("prereqs", []),
                "has_page": kc["id"] in pages,
                "drills_tagged": counts.get(kc["id"], 0),
            }
        )
    return {
        "lessons": registry["lessons"],
        "kc_count": len(registry["kcs"]),
        "kcs": nodes,
        "problems": check(registry),
    }


def read_node(kc_id: str) -> dict:
    registry = load()
    known = _by_id(registry)
    if kc_id not in known:
        raise KeyError(f"No KC '{kc_id}' in the registry.")
    node = dict(known[kc_id])
    node["position"] = [k["id"] for k in registry["kcs"]].index(kc_id)
    node["dependents"] = [k["id"] for k in registry["kcs"] if kc_id in k.get("prereqs", [])]
    node["has_page"] = kc_id in lessons.index()
    node["drills_tagged"] = _drill_counts().get(kc_id, 0)
    return node


def add_kc(
    kc_id: str,
    lesson: str,
    title: str,
    prereqs: list[str] | None = None,
    syntax: bool = True,
    after: str | None = None,
) -> dict:
    """Insert a concept. Default position is straight after its last prereq."""
    registry = load()
    known = _by_id(registry)
    if kc_id in known:
        raise ValueError(f"KC '{kc_id}' already exists — use graph_update_kc.")
    prereqs = list(prereqs or [])
    order = [k["id"] for k in registry["kcs"]]
    for prereq in prereqs:
        if prereq not in known:
            raise ValueError(f"Unknown prereq '{prereq}'.")

    if after:
        if after not in order:
            raise ValueError(f"Unknown anchor '{after}' for `after`.")
        position = order.index(after) + 1
    elif prereqs:
        position = max(order.index(p) for p in prereqs) + 1
    else:
        position = len(order)

    node = {"id": kc_id, "lesson": lesson, "title": title, "syntax": bool(syntax), "prereqs": prereqs}
    registry["kcs"].insert(position, node)
    problems = check(registry)
    if problems:
        raise ValueError("Refusing to write a broken registry: " + "; ".join(problems))
    _save(registry)
    return {"added": kc_id, "position": position, "node": node}


def update_kc(kc_id: str, title: str | None = None, prereqs: list[str] | None = None,
              lesson: str | None = None, after: str | None = None) -> dict:
    registry = load()
    order = [k["id"] for k in registry["kcs"]]
    if kc_id not in order:
        raise KeyError(f"No KC '{kc_id}' in the registry.")
    node = registry["kcs"][order.index(kc_id)]
    if title is not None:
        node["title"] = title
    if lesson is not None:
        node["lesson"] = lesson
    if prereqs is not None:
        node["prereqs"] = list(prereqs)
    if after is not None:
        registry["kcs"].remove(node)
        fresh = [k["id"] for k in registry["kcs"]]
        if after not in fresh:
            raise ValueError(f"Unknown anchor '{after}'.")
        registry["kcs"].insert(fresh.index(after) + 1, node)
    problems = check(registry)
    if problems:
        raise ValueError("Refusing to write a broken registry: " + "; ".join(problems))
    _save(registry)
    return {"updated": kc_id, "node": node}
