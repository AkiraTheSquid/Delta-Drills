"""Behavioral checks for source-derived curriculum navigation."""
import json
import re
import unittest
from pathlib import Path

from arena_curriculum import annotate
from compile_arena_notebooks import _cells

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Local_Deployed_Shared/lessons/notebooks"


class CurriculumTest(unittest.TestCase):
    def test_empty_answer_and_solution_disclosure(self):
        raw = {"cells": [
            {"cell_type": "markdown", "source": ["## Setup"]},
            {"cell_type": "code", "source": ["import math"]},
            {"cell_type": "markdown", "source": ["### Exercise - explain the shape"]},
            {"cell_type": "code", "source": []},
            {"cell_type": "markdown", "source": ["<details><summary>Solution</summary>\n\n### Exercise - secret\n\n```python\nanswer = 42\n```\n</details>"]},
        ]}
        cells = _cells(raw, "test")
        meta = annotate(cells, "test")
        self.assertEqual(meta["setup_cells"], ["test-c001"])
        self.assertEqual(len(meta["exercises"]), 1)
        self.assertEqual(meta["exercises"][0]["answer_cell"], "test-c003")
        self.assertEqual(meta["exercises"][0]["topic_title"], "Setup")
        self.assertFalse(next(c for c in cells if c["id"] == "test-c003")["src"])

    def test_standalone_tasks_keep_their_prompt_and_faded_answers(self):
        cells = [
            {"id": "p", "t": "md", "role": "prose", "src": "## Build a ray"},
            {"id": "demo", "t": "code", "role": "code", "src": "print(42)"},
            {"id": "answer", "t": "code", "role": "code", "src": "# Faded 1.1 — fill the shape\nx = t.zeros(...)"},
        ]
        ex = annotate(cells, "test")["exercises"][0]
        self.assertEqual(ex["prompt_cell"], "answer")
        self.assertEqual(ex["answer_cell"], "answer")
        cells[-1]["src"] = "# Your code here"
        self.assertEqual(annotate(cells, "test")["exercises"][0]["prompt_cell"], "p")

    def test_reasoning_task_gets_answer_without_revealing_solution(self):
        cells = [
            {"id": "p", "t": "md", "role": "prose", "src": "### Exercise - reason about shapes"},
            {"id": "s", "t": "md", "role": "details", "src": "The answer is 42."},
            {"id": "next", "t": "md", "role": "prose", "src": "## Continue"},
        ]
        ex = annotate(cells, "x")["exercises"][0]
        self.assertEqual(ex["answer_cell"], "p-answer")
        self.assertEqual(cells[1]["src"], "# Write your reasoning here.")
        self.assertEqual(cells[2]["role"], "details")

    def test_whole_book_has_working_exercise_links(self):
        index = json.loads((OUT / "arena-index.json").read_text())["sections"]
        book = (ROOT / "arena-book/_toc.yml").read_text()
        numbers = set(re.findall(r'title: "([\d.]+) ', book))
        self.assertEqual(numbers, {s["number"] for s in index})
        self.assertEqual(len({s["chapter"] for s in index}), 5)
        graph = json.loads((OUT / "arena-curriculum.json").read_text())
        ids = {n["id"] for n in graph["nodes"]}
        self.assertEqual(len(ids), len(graph["nodes"]))
        self.assertTrue(all(e["source"] in ids and e["target"] in ids for e in graph["edges"]))
        linked = {n["id"] for n in graph["nodes"] if n["kind"] == "exercise"}
        topics = {n["id"] for n in graph["nodes"] if n["kind"] == "topic"}
        contextualized = {
            e["target"] for e in graph["edges"]
            if e["relation"] == "topic_exercise" and e["source"] in topics
        }
        expected = set()
        for section in index:
            nb = json.loads((OUT / section["file"]).read_text())
            cells = {c["id"]: c for c in nb["cells"]}
            self.assertEqual(len(cells), len(nb["cells"]))
            self.assertTrue(nb["setup_cells"], section["id"])
            self.assertTrue(all(cells[i]["t"] == "code" for i in nb["setup_cells"]))
            for ex in nb["exercises"]:
                expected.add(ex["id"])
                self.assertIn(ex["prompt_cell"], cells)
                self.assertEqual(cells[ex["answer_cell"]]["t"], "code")
                self.assertNotIn(ex["answer_cell"], nb["setup_cells"])
        self.assertEqual(linked, expected)
        self.assertEqual(contextualized, expected)
        self.assertGreater(len(topics), 100)
        self.assertGreater(len(linked), 400)


if __name__ == "__main__":
    unittest.main()
