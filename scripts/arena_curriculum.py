"""Source-derived ARENA navigation, separate from the graded KC registry.

Sequence edges describe reading order, never mastery prerequisites. Exercise
prompts and answers stay in their original notebooks; this index stores links.
"""
from __future__ import annotations

import re
from pathlib import Path
import json

HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.M)
EXERCISE = re.compile(r"^(?:Bonus\s+)?Exercise(?:\s|:|$|\()", re.I)
PLACEHOLDER = re.compile(r"your code here|YOUR_CODE_HERE|NotImplementedError|#\s*TODO", re.I)
PRACTICE = re.compile(r"^#\s*(?:Faded|Challenge|Independent|Integrated|Guided|Solo)\s+\d", re.M | re.I)


def annotate(cells: list[dict], slug: str) -> dict:
    """Find genuine exercise headings; retain unheaded prerequisite tasks too."""
    exercises = []
    active = None
    setup_ids = []
    setup_depth = None
    first_exercise = False
    prose = None
    headings = []

    def context() -> dict:
        """Nearest authored lesson heading, excluding exercise headings."""
        if not headings:
            return {"topic_id": f"topic:{slug}", "topic_title": "Section overview",
                    "topic_cell": cells[0]["id"]}
        _, title, cell_id = headings[-1]
        return {"topic_id": f"topic:{cell_id}", "topic_title": title,
                "topic_cell": cell_id}

    for cell in cells:
        if cell["role"] == "details":
            continue  # A solution disclosure cannot create an exercise or setup.
        if cell["t"] == "md":
            prose = cell["id"]
            for marks, title in HEADING.findall(cell["src"]):
                depth = len(marks)
                while headings and headings[-1][0] >= depth:
                    headings.pop()
                if setup_depth is not None and depth <= setup_depth:
                    setup_depth = None
                if not first_exercise and re.match(r"^Setup(?:\s|$|\()", title, re.I):
                    setup_depth = depth
                if active and depth <= active["depth"]:
                    active = None
                if EXERCISE.match(title):
                    first_exercise = True
                    setup_depth = None
                    active = {
                        "id": f"exercise:{cell['id']}", "title": title,
                        "prompt_cell": cell["id"], "answer_cells": [],
                        "depth": depth, "section": slug, **context(),
                    }
                    exercises.append(active)
                else:
                    headings.append((depth, title, cell["id"]))
            continue
        if setup_depth is not None and not PLACEHOLDER.search(cell["src"]):
            setup_ids.append(cell["id"])
        if active:
            active["answer_cells"].append(cell["id"])
        elif PLACEHOLDER.search(cell["src"]) or PRACTICE.search(cell["src"]):
            # Prerequisites group many independent tasks below one heading.
            # A stable answer-cell address distinguishes those tasks.
            fn = re.search(r"(?:def|class)\s+(\w+)", cell["src"])
            comment = re.search(r"^#\s*(.+)", cell["src"], re.M)
            title = fn[1] if fn else (comment[1] if comment else "Practice")
            exercises.append({"id": f"exercise:{cell['id']}", "title": title,
                              "prompt_cell": cell["id"] if PRACTICE.search(cell["src"]) else (prose or cell["id"]),
                              "answer_cells": [cell["id"]], "section": slug,
                              **context()})

    # This notebook has no Setup heading; these are its two upstream setup
    # cells, not a heuristic that would auto-run arbitrary opening examples.
    if slug == "3-4" and not setup_ids:
        setup_ids = [c["id"] for c in cells if c["id"] in {"3-4-c005", "3-4-c006"}]
    # Data and helper definitions explicitly supplied outside the opening
    # Setup section. Keep these visible in their original reading position.
    extras = {"0-0": {"0-0-c029"}, "0-1": {"0-1-c020"}}
    setup_ids.extend(c["id"] for c in cells if c["id"] in extras.get(slug, set()))
    for ex in exercises:
        ex.pop("depth", None)
        # Prefer the learner's stub over a supplied demonstration/test cell.
        answers = [c for c in cells if c["id"] in ex["answer_cells"]]
        stub = next((c for c in answers if PLACEHOLDER.search(c["src"]) or not c["src"].strip()), None)
        ex["answer_cell"] = stub["id"] if stub else (ex["answer_cells"][0] if answers else None)
        if ex["answer_cell"] is None:
            # Written-reasoning tasks also need somewhere to answer. Comments
            # are editable Python, without pretending to automatically grade prose.
            answer = {"id": f"{ex['prompt_cell']}-answer", "t": "code", "role": "code",
                      "src": "# Write your reasoning here.", "generated_answer": True}
            position = next(i for i, c in enumerate(cells) if c["id"] == ex["prompt_cell"])
            cells.insert(position + 1, answer)
            ex["answer_cell"] = answer["id"]
            ex["answer_cells"] = [answer["id"]]
    return {"setup_cells": setup_ids, "exercises": exercises}


def curriculum(notebooks: list[dict], repo: Path) -> dict:
    """Combine source navigation with existing, explicitly authored prep links."""
    shared = repo / "Local_Deployed_Shared"
    registry = json.loads((shared / "lessons/kc_registry.json").read_text())
    practice = json.loads((shared / "lessons/arena_exercise_kcs.json").read_text())
    vocab = json.loads((repo / "concept-graph/vocab/atoms.json").read_text())
    atoms = {a["id"]: a for a in vocab["atoms"]}
    annotations = [json.loads(p.read_text()) for p in sorted((repo / "concept-graph/exercises").glob("*.json"))]
    nodes, edges = [], []
    def edge(source, target, relation):
        edges.append({"source": source, "target": target, "relation": relation})
    for kc in registry["kcs"]:
        nodes.append({"id": f"kc:{kc['id']}", "kind": "kc", "title": kc["title"], "kc": kc["id"]})
        for pre in kc.get("prereqs", []):
            edge(f"kc:{pre}", f"kc:{kc['id']}", "prerequisite")
    used_atoms = set()
    used_topics = set()
    previous = {}
    for nb in notebooks:
        sid = f"section:{nb['id']}"
        nodes.append({"id": sid, "kind": "section", "section": nb["id"],
                      "title": f"{nb['number']} {nb['title']}", "chapter": nb["chapter"],
                      "description": nb["desc"], "book_url": nb["book_url"],
                      "exercise_count": len(nb["exercises"])})
        if nb["chapter"] in previous:
            edge(previous[nb["chapter"]], sid, "course_order")
        previous[nb["chapter"]] = sid
        last = sid
        for ex in nb["exercises"]:
            node = {**ex, "kind": "exercise", "chapter": nb["chapter"], "prep_kcs": [], "concepts": []}
            topic_id = ex["topic_id"]
            if topic_id not in used_topics:
                used_topics.add(topic_id)
                nodes.append({"id": topic_id, "kind": "topic", "section": nb["id"],
                              "chapter": nb["chapter"], "title": ex["topic_title"],
                              "cell": ex["topic_cell"]})
                edge(sid, topic_id, "lesson_topic")
            edge(topic_id, ex["id"], "topic_exercise")
            for symbol, mapping in practice.get(nb["id"], {}).items():
                if re.search(r"(?<!\w)" + re.escape(symbol) + r"(?!\w)", ex["title"]):
                    node["prep_kcs"].append(mapping["kc"])
                    edge(f"kc:{mapping['kc']}", ex["id"], "prepares")
            # Exact normalized title + section matching; positional matching
            # would be wrong for the fork's supplementary exercises.
            normalize = lambda text: re.sub(r"[^a-z0-9]+", "", re.sub(r"^Exercise\s*[-:]?\s*", "", text, flags=re.I).lower())
            tagged = next((a for a in annotations if
                           f"{a['arena_chapter']}-{a['arena_part']}" == nb["id"]
                           and normalize(a["title"]) == normalize(ex["title"])), None)
            if tagged:
                for atom in tagged["atoms"]:
                    if atom["id"] not in atoms:
                        continue
                    used_atoms.add(atom["id"])
                    node["concepts"].append({"id": atom["id"], "role": atom["role"]})
                    edge(f"concept:{atom['id']}", ex["id"], "exercises_concept")
            nodes.append(node)
            edge(last, ex["id"], "exercise_order")
            last = ex["id"]
    for aid in sorted(used_atoms):
        a = atoms[aid]
        nodes.append({"id": f"concept:{aid}", "kind": "concept", "title": a["label"],
                      "description": a["definition"]})
    return {"version": 1, "nodes": nodes, "edges": edges,
            "description": "ARENA source exercises and preparation. Order is not mastery; unmapped exercises make no drill-coverage claim."}
