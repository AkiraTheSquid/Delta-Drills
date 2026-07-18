#!/usr/bin/env python3
"""Compile KP markdown sources into Local_Deployed_Shared/lessons/lessons_structured.json.

Usage: python3 scripts/compile_lessons.py
Run scripts/validate_lessons.py first (or with --gate) — compilation does not re-check code.
"""
import json
import sys
from lesson_lib import LESSONS_DIR, all_kp_paths, load_registry, parse_kp, split_items, code_fences


def compile_lessons():
    registry = load_registry()
    kc_by_id = {kc["id"]: kc for kc in registry["kcs"]}
    lessons = {l["id"]: {**l, "kps": []} for l in registry["lessons"]}

    for path in all_kp_paths():
        kp = parse_kp(path)
        kc = kc_by_id[kp["kc"]]
        faded_items = []
        for qid, content in split_items(kp["sections"].get("Faded practice", "")).items():
            starters = code_fences(content, "python starter")
            solutions = code_fences(content, "python solution")
            prose = content.split("```", 1)[0].strip()
            faded_items.append({
                "question_id": qid,
                "prompt": prose,
                "starter_code": starters[0] if starters else "",
                "solution": solutions[0] if solutions else "",
            })
        guided_items = []
        for qid, content in split_items(kp["sections"].get("Guided practice", "")).items():
            guided_items.append({"question_id": qid, "hints_markdown": content})
        lessons[kc["lesson"]]["kps"].append({
            "kc": kp["kc"],
            "title": kp["title"] or kc["title"],
            "supporting_kcs": kp["supporting"],
            "new_syntax": kp["new_syntax"],
            "concept_markdown": kp["sections"].get("Concept", ""),
            "worked_example_markdown": kp["sections"].get("Worked example", ""),
            "faded_items": faded_items,
            "guided_items": guided_items,
            "independent_items": kp["independent"],
            "misconceptions_markdown": kp["sections"].get("Misconceptions", ""),
        })

    # Order KPs within each lesson by registry order (authoring source of truth).
    kc_order = {kc["id"]: i for i, kc in enumerate(registry["kcs"])}
    for lesson in lessons.values():
        lesson["kps"].sort(key=lambda kp: kc_order[kp["kc"]])

    out = {
        "version": registry["version"],
        "lessons": [lessons[l["id"]] for l in registry["lessons"]],
    }
    out_path = LESSONS_DIR / "lessons_structured.json"
    out_path.write_text(json.dumps(out, indent=1))
    n_kps = sum(len(l["kps"]) for l in lessons.values())
    print(f"compiled {n_kps} KPs across {len(lessons)} lessons -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(compile_lessons())
