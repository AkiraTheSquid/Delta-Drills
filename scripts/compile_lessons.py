#!/usr/bin/env python3
"""Compile KP markdown sources into Local_Deployed_Shared/lessons/lessons_structured.json.

Usage: python3 scripts/compile_lessons.py
Run scripts/validate_lessons.py first (or with --gate) — compilation does not re-check code.
"""
import json
import re
import sys
from lesson_lib import (
    LESSONS_DIR,
    all_kp_paths,
    attach_example_run,
    code_fences,
    load_bank,
    load_registry,
    parse_kp,
    split_items,
)


def compile_lessons():
    registry = load_registry()
    bank = load_bank()
    kc_by_id = {kc["id"]: kc for kc in registry["kcs"]}
    lessons = {l["id"]: {**l, "kps": []} for l in registry["lessons"]}

    for path in all_kp_paths():
        kp = parse_kp(path)
        kc = kc_by_id[kp["kc"]]

        def _faded_items(section_text):
            items = []
            for qid, content in split_items(section_text).items():
                starters = code_fences(content, "python starter")
                solutions = code_fences(content, "python solution")
                prose = content.split("```", 1)[0].strip()
                # The authored starter is the function alone. Its question's own
                # starter ends with a fixture and a print, and without that the
                # learner has nothing to compare the expected output against —
                # so graft it on here, once, for every route that serves this
                # record (see lesson_lib.attach_example_run).
                exercise = (bank.get(qid) or {}).get("exercise") or {}
                items.append({
                    "question_id": qid,
                    "prompt": prose,
                    "starter_code": attach_example_run(
                        starters[0] if starters else "",
                        exercise.get("starter_code") or "",
                        exercise.get("function_name") or "solve",
                    ),
                    "solution": solutions[0] if solutions else "",
                })
            return items

        # One page per single-concept segment (the in-app player steps
        # through these); legacy aggregate fields kept for viewer.html.
        segments = [{
            "concept_id": kp["concepts"][i] if i < len(kp["concepts"]) else "",
            "title": seg["title"],
            "concept_markdown": seg["concept"],
            "watch_out_markdown": seg["watch_out"],
            "worked_example_markdown": seg["worked"],
            "worked_example_code": code_fences(seg["worked"], "python")[0],
            "faded_items": _faded_items(seg["faded"]),
        } for i, seg in enumerate(kp["segments"])]
        faded_items = [item for seg in segments for item in seg["faded_items"]]
        guided_items = []
        for qid, content in split_items(kp["sections"].get("Guided practice", "")).items():
            # A guided drill is served at the same ladder stage as a faded one
            # (kc_graph._STAGE_TO_RANKS puts both on the supported rungs), so it
            # needs the same thing above it: a solved example of the move, not
            # only hints hidden behind a <details>. An optional ```python worked
            # fence carries it. It is split OUT of the hints so the example is
            # visible without opening anything — the whole point is that the
            # learner reads it first.
            worked = code_fences(content, "python worked")
            hints = re.sub(
                r"```python worked\n.*?```\n?", "", content, flags=re.S
            ).strip()
            guided_items.append({
                "question_id": qid,
                "hints_markdown": hints,
                "worked_example_code": worked[0] if worked else "",
            })
        # Applied practice is the THIRD rung of the ladder: the drill is an
        # independent one — no blanks, the learner writes the whole function —
        # but an example of the same move sits above it. That is what separates
        # it from `solo`, where the same kind of drill arrives with nothing to
        # read first. The question ids stay in the `independent` frontmatter
        # list, so the rung the backend derives is unchanged; what this section
        # adds is the example, and having one is what routes the drill to
        # `partial` rather than `solo` (see kc_graph.questions_at_stage).
        applied_items = []
        for qid, content in split_items(kp["sections"].get("Applied practice", "")).items():
            worked = code_fences(content, "python worked")
            applied_items.append({
                "question_id": qid,
                "prompt": content.split("```", 1)[0].strip(),
                "worked_example_code": worked[0] if worked else "",
            })
        lessons[kc["lesson"]]["kps"].append({
            "kc": kp["kc"],
            "title": kp["title"] or kc["title"],
            "supporting_kcs": kp["supporting"],
            "new_syntax": kp["new_syntax"],
            "concept_markdown": kp["sections"].get("Concept", ""),
            "worked_example_markdown": kp["sections"].get("Worked example", ""),
            "segments": segments,
            "faded_items": faded_items,
            "guided_items": guided_items,
            "applied_items": applied_items,
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
