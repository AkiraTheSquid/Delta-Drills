"""The math (multiple-choice) lane's gate, on synthetic fixtures — no content.

Run: This-Directory-Only/backend/.venv/bin/python3 scripts/test_validate_math.py
(any python with sympy works). Spec: docs/spec-math-mc-backbone.md.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import validate_math as V
from lesson_lib import parse_kp

REGISTRY = {
    "version": 1,
    "lessons": [{"id": "ma-1", "topic": "Mathematics", "title": "T", "subtopic_key": "Mathematics: T"},
                {"id": "tr-1", "topic": "PyTorch", "title": "R", "subtopic_key": "PyTorch: R"}],
    "kcs": [{"id": "math.dot", "lesson": "ma-1", "title": "Dot", "syntax": False, "prereqs": []},
            {"id": "torch.x", "lesson": "tr-1", "title": "X", "syntax": True, "prereqs": []}],
}

COMPUTE = {
    "id": 50001, "kind": "compute", "difficulty": 30,
    "prompt": "Compute $\\vec a \\cdot \\vec b$ for $a=(1,2,3)$, $b=(4,5,6)$.",
    "choices": [
        {"key": "A", "text": "$32$", "value": "32"},
        {"key": "B", "text": "$-32$", "value": "-32"},
        {"key": "C", "text": "$21$", "value": "21"},
    ],
    "answer": "A",
    "solution": "$1\\cdot4 + 2\\cdot5 + 3\\cdot6 = 32$.",
    "verify": {"sympy": {"truth": "Matrix([1,2,3]).dot(Matrix([4,5,6]))"}},
}
DERIVATION = {
    "id": 50002, "kind": "derivation-step", "difficulty": 45,
    "prompt": "Given $\\|a-b\\|^2 = (a-b)\\cdot(a-b)$, which line follows?",
    "choices": [
        {"key": "A", "text": "$a\\cdot a - 2a\\cdot b + b\\cdot b$", "value": "a**2 - 2*a*b + b**2"},
        {"key": "B", "text": "$a\\cdot a + b\\cdot b$", "value": "a**2 + b**2"},
    ],
    "answer": "A",
    "solution": "Expand the product.",
    "verify": {"sympy": {"symbols": "a b", "truth": "(a-b)*(a-b)"}},
}
STATEMENT = {
    "id": 50003, "kind": "statement", "difficulty": 20,
    "prompt": "Which is true of an orthogonal matrix $Q$?",
    "choices": [{"key": "A", "text": "$Q^TQ = I$"}, {"key": "B", "text": "$\\det Q = 0$"}],
    "answer": "A",
    "solution": "By definition.",
}
KP_MD = """---
kc: math.dot
kind: math
title: Dot product
supporting: []
new_syntax: []
previews: []
concepts: [dot-definition]
faded: []
guided: []
independent: [50001, 50002]
integrated: [50003]
---

## Concept: Dot product
$a \\cdot b = \\sum_i a_i b_i$.

## Worked example
For $a=(1,0)$, $b=(0,1)$: $0$.

## Solo practice
### q50001
Compute it.
### q50002
Next line.

## Integrated practice
### q50003
Which is true.
"""


class SympyFindings(unittest.TestCase):
    def test_compute_verified(self):
        self.assertEqual(V.sympy_findings(COMPUTE, "q"), [])

    def test_derivation_step_verified(self):
        self.assertEqual(V.sympy_findings(DERIVATION, "q"), [])

    def test_wrong_key_fails(self):
        bad = copy.deepcopy(COMPUTE)
        bad["answer"] = "C"
        found = V.sympy_findings(bad, "q50001")
        self.assertTrue(any("keyed choice C" in f for f in found), found)

    def test_distractor_equal_to_truth_fails(self):
        bad = copy.deepcopy(COMPUTE)
        bad["choices"][1]["value"] = "16*2"
        found = V.sympy_findings(bad, "q50001")
        self.assertTrue(any("distractor B" in f and "two correct" in f for f in found), found)

    def test_missing_value_fails(self):
        bad = copy.deepcopy(COMPUTE)
        del bad["choices"][2]["value"]
        self.assertTrue(any("choice C has no `value`" in f for f in V.sympy_findings(bad, "q")))

    def test_statement_needs_no_sympy_and_refuses_it(self):
        self.assertEqual(V.sympy_findings(STATEMENT, "q"), [])
        bad = dict(STATEMENT, verify={"sympy": {"truth": "1"}})
        self.assertTrue(V.sympy_findings(bad, "q"))

    def test_empty_statement_sympy_block_refused(self):
        bad = dict(STATEMENT, verify={"sympy": {}})
        self.assertTrue(V.sympy_findings(bad, "q"))

    def test_malformed_choice_is_a_finding_not_a_crash(self):
        bad = copy.deepcopy(COMPUTE)
        bad["choices"].append("D. 40")
        self.assertTrue(V.problem_findings(bad, "q"))
        self.assertEqual(V.sympy_findings(bad, "q"), [])

    def test_compute_without_truth_fails(self):
        bad = dict(COMPUTE, verify={})
        self.assertTrue(any("needs verify.sympy.truth" in f for f in V.sympy_findings(bad, "q")))

    def test_matrix_shape_mismatch_is_a_difference(self):
        prob = copy.deepcopy(COMPUTE)
        prob["verify"] = {"sympy": {"truth": "Matrix([[32]])"}}
        found = V.sympy_findings(prob, "q")
        self.assertTrue(any("keyed choice A" in f for f in found), found)

    def test_symbolic_relation_compares_without_crashing(self):
        # bool(Eq(x, 1)) raises TypeError in SymPy; a Boolean truth must be a
        # finding or a pass, never a traceback.
        prob = copy.deepcopy(COMPUTE)
        prob["verify"] = {"sympy": {"symbols": "x", "truth": "Eq(x, 1)"}}
        prob["choices"] = [
            {"key": "A", "text": "$x = 1$", "value": "Eq(1, x)"},
            {"key": "B", "text": "$x = 2$", "value": "Eq(x, 2)"},
            {"key": "C", "text": "$1$", "value": "1"},
        ]
        self.assertEqual(V.sympy_findings(prob, "q"), [])
        prob["answer"] = "B"
        found = V.sympy_findings(prob, "q")
        self.assertTrue(any("keyed choice B" in f for f in found), found)
        self.assertTrue(any("distractor A" in f for f in found), found)

    def test_unparseable_truth_is_an_error_not_a_pass(self):
        prob = dict(COMPUTE, verify={"sympy": {"truth": "Matrix(["}})
        self.assertTrue(any("does not evaluate" in f for f in V.sympy_findings(prob, "q")))


class ProblemShape(unittest.TestCase):
    def test_clean(self):
        for p in (COMPUTE, DERIVATION, STATEMENT):
            self.assertEqual(V.problem_findings(p, "q"), [])

    def test_id_floor_and_keys_and_answer(self):
        bad = copy.deepcopy(COMPUTE)
        bad["id"] = 12
        bad["choices"][1]["key"] = "Z"
        bad["answer"] = "Q"
        found = V.problem_findings(bad, "q")
        self.assertEqual(len(found), 3, found)

    def test_duplicate_text(self):
        bad = copy.deepcopy(COMPUTE)
        bad["choices"][1]["text"] = bad["choices"][0]["text"]
        self.assertTrue(any("same text" in f for f in V.problem_findings(bad, "q")))

    def test_difficulty_range_and_kind(self):
        bad = dict(COMPUTE, difficulty=5, kind="proof")
        found = V.problem_findings(bad, "q")
        self.assertTrue(any("difficulty" in f for f in found) and any("kind" in f for f in found))


class ProblemFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.md = self.dir / "kp-dot.md"
        self.md.write_text(KP_MD)
        self.problems = self.dir / "kp-dot.problems.json"
        self.problems.write_text(json.dumps({"kc": "math.dot", "problems": [COMPUTE, DERIVATION, STATEMENT]}))
        self.atoms = {50001: ["dot-product"], 50002: ["dot-product"], 50003: ["orthogonal"]}
        self.nodes = {"dot-product", "orthogonal"}

    def tearDown(self):
        self.tmp.cleanup()

    def run_check(self, **overrides):
        errors = []
        V.check_problem_file(
            self.problems, overrides.get("registry", REGISTRY), {},
            overrides.get("csv_ids", set()), overrides.get("atoms", self.atoms),
            overrides.get("nodes", self.nodes), errors,
        )
        return errors

    def test_clean_file_passes(self):
        self.assertEqual(self.run_check(), [])

    def test_missing_atom_row_named_by_id(self):
        errors = self.run_check(atoms={50001: ["dot-product"], 50002: ["dot-product"]})
        self.assertTrue(any("q50003" in e and "question_atom_tags" in e for e in errors), errors)

    def test_atom_not_in_graph(self):
        errors = self.run_check(nodes={"dot-product"})
        self.assertTrue(any("orthogonal" in e and "not a node" in e for e in errors), errors)

    def test_csv_collision(self):
        errors = self.run_check(csv_ids={50002})
        self.assertTrue(any("q50002" in e and "collides" in e for e in errors), errors)

    def test_wrong_lesson_topic(self):
        reg = copy.deepcopy(REGISTRY)
        reg["kcs"][0]["lesson"] = "tr-1"
        errors = self.run_check(registry=reg)
        self.assertTrue(any("not one of" in e and "Mathematics" in e for e in errors), errors)

    def test_problem_on_no_rung_and_rung_without_problem(self):
        self.md.write_text(KP_MD.replace("independent: [50001, 50002]", "independent: [50001, 50009]"))
        errors = self.run_check()
        self.assertTrue(any("on no rung" in e and "50002" in e for e in errors), errors)
        self.assertTrue(any("no problem" in e and "50009" in e for e in errors), errors)

    def test_problem_on_two_rungs_refused(self):
        self.md.write_text(KP_MD.replace("integrated: [50003]", "integrated: [50003, 50001]"))
        errors = self.run_check()
        self.assertTrue(any("more than one rung" in e and "50001" in e for e in errors), errors)

    def test_targeted_run_sees_other_files_ids(self):
        other = self.dir / "kp-other.problems.json"
        other.write_text(json.dumps({"kc": "math.other", "problems": [dict(COMPUTE, id=50001)]}))
        (self.dir / "kp-other.md").write_text(KP_MD)
        old_files = V.math_bank.problem_files
        V.math_bank.problem_files = lambda _dir=None: [self.problems, other]
        try:
            errors = []
            V.check_all(errors, [self.problems])
        finally:
            V.math_bank.problem_files = old_files
        self.assertTrue(any("q50001" in e and "already used in kp-other" in e for e in errors), errors)

    def test_guided_rung_refused(self):
        self.md.write_text(KP_MD.replace("guided: []", "guided: [50001]").replace("independent: [50001, 50002]", "independent: [50002]"))
        errors = self.run_check()
        self.assertTrue(any("legacy rung" in e for e in errors), errors)

    def test_kp_without_kind_math(self):
        self.md.write_text(KP_MD.replace("kind: math\n", ""))
        errors = self.run_check()
        self.assertTrue(any("kind: math" in e for e in errors), errors)


class MathKp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.md = self.dir / "kp-dot.md"
        (self.dir / "kp-dot.problems.json").write_text("{}")
        self.bank = {
            50001: {"exercise": {"submission_mode": "mc"}},
            50002: {"exercise": {"submission_mode": "mc"}},
            50003: {"exercise": {"submission_mode": "mc"}},
        }

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, text):
        self.md.write_text(text)
        errors = []
        V.check_math_kp(parse_kp(self.md), self.md, self.bank, errors)
        return errors

    def test_clean(self):
        self.assertEqual(self.check(KP_MD), [])

    def test_unchecked_runnable_fence_refused(self):
        text = KP_MD.replace("## Worked example\n", "## Worked example\n```python\nprint(1)\n```\n")
        self.assertTrue(any("no assertion" in e for e in self.check(text)))

    def test_checked_demonstration_allowed(self):
        text = KP_MD.replace("## Worked example\n", "## Worked example\n```python\nx = 2 + 3\nprint(x)\n# Hidden checks\nassert x == 5\n```\n")
        self.assertEqual(self.check(text), [])

    def test_wrong_demonstration_refused(self):
        text = KP_MD.replace("## Worked example\n", "## Worked example\n```python\nx = 2 + 3\n# Hidden checks\nassert x == 6\n```\n")
        self.assertTrue(self.check(text))

    def test_no_run_fence_allowed(self):
        text = KP_MD.replace("## Worked example\n", "## Worked example\n```python no-run\nt.dot(a, b)\n```\n")
        self.assertEqual(self.check(text), [])

    def test_two_segments_need_carriers(self):
        text = KP_MD.replace("concepts: [dot-definition]", "concepts: [dot-definition, dot-geometry]")
        text = text.replace("## Solo practice", "## Concept: Geometry\nAngle.\n\n## Worked example\nCos.\n\n## Solo practice")
        errors = self.check(text)
        self.assertEqual(sum("Faded practice" in e for e in errors), 2, errors)

    def test_same_id_under_two_sections_refused(self):
        text = KP_MD.replace("integrated: [50003]", "integrated: [50003, 50001]")
        text = text.replace("## Integrated practice\n### q50003", "## Integrated practice\n### q50001\nAgain.\n### q50003")
        self.assertTrue(any("both Solo and Integrated" in e for e in self.check(text)))

    def test_duplicate_heading_in_one_section_refused(self):
        text = KP_MD.replace("### q50002\nNext line.", "### q50002\nNext line.\n### q50002\nAgain.")
        self.assertTrue(any("more than once" in e and "50002" in e for e in self.check(text)))

    def test_indented_runnable_fence_refused(self):
        text = KP_MD.replace("## Worked example\n", "## Worked example\n  ```python\n  print(1)\n  ```\n")
        self.assertTrue(any("no assertion" in e for e in self.check(text)))

    def test_coding_drill_refused_on_math_page(self):
        self.bank[50001] = {"exercise": {"submission_mode": "function"}}
        self.assertTrue(any("coding drill" in e for e in self.check(KP_MD)))

    def test_missing_from_bank_says_export(self):
        del self.bank[50003]
        self.assertTrue(any("export_questions_json" in e for e in self.check(KP_MD)))


class SharedLoader(unittest.TestCase):
    def test_flat_row_shape(self):
        import math_bank

        row = math_bank.flat_row(COMPUTE, "math.dot", {"topic": "Mathematics", "subtopic": "T", "subtopic_key": "Mathematics: T"}, "x")
        self.assertEqual(row["submission_mode"], "mc")
        self.assertEqual(row["expected_output"], "A")
        self.assertEqual(row["correct_choice"], "A")
        self.assertEqual(row["choices"][0], {"key": "A", "text": "$32$", "value": "32"})
        self.assertEqual(row["subtopic_key"], "Mathematics: T")
        self.assertEqual(row["answer_code"], "")
        self.assertEqual(row["test_cases"], [])
        self.assertEqual(row["difficulty_label"], "easy")


    def test_duplicate_id_across_files_raises(self):
        import math_bank

        with tempfile.TemporaryDirectory() as tmp:
            lessons = Path(tmp) / "lessons"
            (lessons / "mathematics").mkdir(parents=True)
            (lessons / "kc_registry.json").write_text(json.dumps(REGISTRY))
            for slug in ("kp-a", "kp-b"):
                (lessons / "mathematics" / f"{slug}.problems.json").write_text(
                    json.dumps({"kc": "math.dot", "problems": [COMPUTE]}))
            with self.assertRaises(ValueError) as ctx:
                math_bank.load_math_rows(lessons)
            self.assertIn("50001", str(ctx.exception))
            self.assertIn("kp-a", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
