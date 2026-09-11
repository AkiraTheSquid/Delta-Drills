"""Regression checks for q1313–1332, including plausible wrong solutions.

Run with the backend venv. References, stubs, constants, type mistakes and
concept-specific near misses all go through the production grading harness.
"""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Local_Deployed_Shared' / 'pipeline'))
from audit_question_bank import load_code_runner
from content_safety import leak_findings
from audit_solution_prereqs import find
from colab_cells import grader_source
from validate_lessons import grade_against_bank


NEAR_MISSES = {
    1313: 'return (x.sum() ** 2).item()',
    1314: 'return (x != 0).sum().item()',
    1315: 'return ((x >= threshold).sum() / x.numel()).item()',
    1316: 'return x.mean().item()',
    1317: 'return ((x - x.mean()) ** 2).sum().item()',
    1318: 'return ((predicted - actual).mean() ** 2).item()',
    1319: 'return ((a * b) <= 0).any().item()',
    1320: 'return (a.mean() - b.mean()).item()',
    1321: 'return (x >= x.mean()).sum().item()',
    1322: 'return 1.0 / x.numel()',
    1323: 'return x.sum().item()',
    1324: 'return t.sqrt(x.sum() ** 2 / x.numel()).item()',
    1325: 'return (x * weights).mean().item()',
    1326: 'return ((a.mean() + b.mean()) / 2).item()',
    1327: 'return x / x.max(), x.sum().item()',
    1328: 'return low + x / x.max() * (high - low), (x.max() - x.min()).item()',
    1329: 'return (target.sum() / x.sum()).item()',
    1330: 'return (target - measured).mean().item(), ((measured - target) ** 2).mean().item()',
    1331: 'return (weights * (x - x.mean()) ** 2).mean().item()',
    1332: 'return (((predicted - actual) ** 2 < tolerance ** 2).sum() / predicted.numel()).item(), ((predicted - actual) ** 2).mean().item()',
}


class AggregationDrillsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = {q['id']: q for q in json.loads((ROOT / 'Local_Deployed_Shared/questions.json').read_text())}
        cls.runner = load_code_runner()

    def passes(self, code, cases):
        results, execution = self.runner.run_function_tests(code, cases)
        return execution.success and len(results) == len(cases) and all(r.passed for r in results)

    def test_real_harness_and_near_misses(self):
        for qid, wrong_body in NEAR_MISSES.items():
            with self.subTest(qid=qid):
                q = self.bank[qid]
                cases = q['test_cases']
                self.assertEqual(len(cases), 4)
                self.assertTrue(self.passes(q['answer_code'], cases), qid)
                self.assertFalse(self.passes(q['starter_code'], cases), qid)
                self.assertFalse(self.passes(q['starter_code'].replace('pass', wrong_body), cases), qid)
                # Returning the first case's correct output must fail elsewhere.
                constant = q['starter_code'].replace('pass', 'return ' + cases[0]['expected_expr'])
                self.assertFalse(self.passes(constant, cases), qid)
                fn = next(n for n in ast.parse(q['starter_code']).body if isinstance(n, ast.FunctionDef))
                noop = q['starter_code'].replace('pass', 'return ' + fn.args.args[0].arg)
                self.assertFalse(self.passes(noop, cases), qid)

    def test_python_scalar_boundary_is_graded(self):
        for qid in NEAR_MISSES:
            with self.subTest(qid=qid):
                q = self.bank[qid]
                tensor_return = q['answer_code'].replace('.item()', '')
                self.assertNotEqual(tensor_return, q['answer_code'])
                self.assertFalse(self.passes(tensor_return, q['test_cases']))

    def test_hidden_assertions_across_all_graders(self):
        # Ordinary numeric equality accepts both 3 and 3.0. Only the hidden
        # assertion distinguishes the correct Python return type here.
        cases = [{'setup_code': 'x = 3', 'call': 'solve(x)', 'expected_expr': '3.0',
                  'assert_code': 'assert type(result) is float'}]
        source = (ROOT / 'Local_Deployed_Shared/practice/api.js').read_text()
        harness = source.split('const resultJson = pyodide.runPython(`', 1)[1].split('`);', 1)[0]
        harness = harness.replace('${testsJsonLiteral}', json.dumps(json.dumps(cases)))
        for body, wanted in [('return float(x)', True), ('return x', False)]:
            solution = 'def solve(x):\n    ' + body
            self.assertEqual(self.passes(solution, cases), wanted)
            self.assertEqual(not grade_against_bank(solution, {'exercise': {'test_cases': cases}}), wanted)
            ns = {}
            exec(grader_source(), ns)
            ns['_DD_TESTS'] = {'1': {'fn': 'solve', 'cases': cases}}
            exec(solution, ns)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(eval('dd_check(1, verbose=False)', ns), wanted)
            ns = {}
            exec(solution, ns)
            exec(harness, ns)
            self.assertEqual(ns['_delta_results'][0]['passed'], wanted)

    def test_input_mutation_is_rejected_before_fixture_reset(self):
        q = self.bank[1327]
        code = q['starter_code'].replace('pass', 'total = x.sum()\n    x /= total\n    return x, total.item()')
        self.assertFalse(self.passes(code, q['test_cases']))

    def test_no_new_syntax_or_solution_leaks(self):
        violations = [v for v in find(('solution', 'starter', 'prompt')) if v['qid'] in NEAR_MISSES]
        self.assertEqual(violations, [])
        for qid in NEAR_MISSES:
            q = self.bank[qid]
            self.assertEqual(leak_findings(q, 'independent' if qid < 1325 else 'integrated', str(qid)), [])

    def test_runtime_rung_registration(self):
        tags = json.loads((ROOT / 'Local_Deployed_Shared/lessons/qmatrix_tags.json').read_text())
        for qid in NEAR_MISSES:
            self.assertEqual(tags[str(qid)]['target_kcs'], ['numpy.aggregations'])
            self.assertEqual(tags[str(qid)]['source'], 'kp-independent' if qid < 1325 else 'kp-integrated')


if __name__ == '__main__':
    unittest.main()
