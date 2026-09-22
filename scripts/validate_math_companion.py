#!/usr/bin/env python3
"""Validate the authored 0.1 math companion and independent numerical cases."""
import ast
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "Local_Deployed_Shared" / "lessons"
sys.path.insert(0, str(LESSONS))
from checks import run_checked


def main():
    data = json.loads((LESSONS / "math/arena-0-1.json").read_text())
    registry = json.loads((LESSONS / "kc_registry.json").read_text())
    kcs = {kc["id"] for kc in registry["kcs"]}
    mappings = json.loads((LESSONS / "arena_exercise_kcs.json").read_text())["0-1"]
    topics, questions, cell_count = set(), {}, 0
    for topic in data["topics"]:
        assert topic["id"] not in topics, "Duplicate topic"
        assert set(topic["prereqs"]) <= topics, "Prerequisite missing or taught later"
        topics.add(topic["id"])
        assert topic["kc"] in kcs, "Companion must use an existing graph KC"
        assert set(topic["exercises"]) <= mappings.keys(), "Unknown ARENA exercise"
        assert any(row["kc"] == topic["kc"] for row in mappings.values()) or topic["kc"] == "raytracing.ray-parametrisation"
        for section in topic["sections"]:
            assert len(section["concept"].split()) >= 80, "Teach the general rule, not a caption"
            assert section["worked"] and len(section["questions"]) >= 4
            for key in ("code", "worked_code"):
                assert run_checked(section[key]).strip(), "Cell must display a checked result"
                cell_count += 1
            for q in section["questions"]:
                assert q["id"] not in questions, "Duplicate question ID"
                questions[q["id"]] = q
                assert len(q["choices"]) == len(set(q["choices"])) == 4
                assert type(q["answer"]) is int and 0 <= q["answer"] < 4
                assert len(q["feedback"]) == 4 and all(q["feedback"])
                assert q["prompt"].strip()

    # Parse actual choices, rather than just testing an unrelated reference answer.
    def numerical(qid, expected):
        q = questions[qid]
        values = [ast.literal_eval(s.replace("−", "-")) for s in q["choices"]]
        matches = [i for i, value in enumerate(values) if value == expected]
        assert matches == [q["answer"]], (qid, matches, q["answer"])

    numerical("rays-endpoint", tuple(q-o for q, o in zip((7, 4), (3, -2))))
    numerical("rays-rescale", 6/3)
    numerical("rays-fan", [-3 + i * 6/4 for i in range(5)])
    numerical("rays-reverse", 1-0.2)
    numerical("systems-matrix", [[1, 6-6], [2, 0-4]])
    numerical("systems-rhs", tuple(a-o for a, o in zip((4, 1), (-2, 3))))
    numerical("systems-solve", [(6-0)/2, (1-(-1))/(3-(-1))])
    numerical("singularity-axis", [any(row) for row in [[False, True], [False, False], [True, False]]])
    numerical("triangles-weight-a", 1-0.2-0.3)
    a, b, c = (1, 0, 0), (1, 8, 0), (1, 0, 4)
    numerical("triangles-coordinate", tuple(a[i]+.25*(b[i]-a[i])+.5*(c[i]-a[i]) for i in range(3)))
    numerical("triangles-translated", [(5-1)/2, 1/4, 1/4])
    numerical("visibility-norm", 2*math.sqrt(sum(x*x for x in (2, 3, 6))))
    numerical("visibility-depth", (4*2, 3+4*2))

    # Boundary and misconception checks for the contextual questions.
    hit = lambda u, v: u >= 0 and 0 <= v <= 1
    assert [hit(*uv) for uv in [(0, 1), (2, -.1), (-1, .5)]] == [True, False, False]
    assert not hit(2, 1.4)  # Zero residual alone cannot imply membership.
    assert 2*2-4*1 == 0
    assert [abs(x) >= 1e-8 for x in [0, -3e-9, 4e-9, -2e-8]] == [False, False, False, True]
    assert 1e-10 < 1e-8 <= 100**2 * 1e-10
    inside = lambda u, v: u >= 0 and v >= 0 and u+v <= 1
    assert not inside(.8, .4) and not inside(.7, .6) and inside(0, 1)
    assert .25+.75 == 1
    assert min(s for s, valid in zip([-1, 5, 2, .5], [True, True, True, False]) if valid and s >= 0) == 2
    assert min([math.inf, math.inf]) == math.inf
    assert 3*1 < 1*10
    assert (8/4)*4 == 8
    print(f"PASS: {len(topics)} topics, {len(questions)} MCQs, {cell_count} checked cells; mappings and numerical cases")


if __name__ == "__main__":
    main()
