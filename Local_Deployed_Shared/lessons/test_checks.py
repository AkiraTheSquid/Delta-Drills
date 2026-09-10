"""Regression tests for authored checks, including the browser's real harness."""
import ast
import json
from pathlib import Path
import subprocess
import unittest

from checks import run_checked, split_checks

HERE = Path(__file__).parent


class LessonChecksTest(unittest.TestCase):
    def test_missing_dormant_and_constant_checks_fail(self):
        for source in ('print(3)', 'def unused():\n    assert 3 == 4',
                       'if False:\n    assert 3 == 4', 'assert True',
                       'x = 3\nassert x == x'):
            with self.subTest(source=source), self.assertRaises(AssertionError):
                run_checked(source)

    def test_checks_validate_the_edited_result(self):
        source = 'x = 2 + 3\nprint(x)\n# Hidden checks\nassert _delta_output == "5\\n"'
        self.assertEqual(run_checked(source), '5\n')
        with self.assertRaises(AssertionError):
            run_checked(source.replace('2 + 3', '2 + 4'))

    def test_echo_and_namespace(self):
        ns = {}
        source = 'x = 5\nx\n# Hidden checks\nhelper = 9\nassert _delta_output == "5\\n"'
        self.assertEqual(run_checked(source, ns), '5\n')
        self.assertNotIn('helper', ns)
        self.assertEqual(run_checked('print(x + 1)\n# Hidden checks\nassert _delta_output == "6\\n"', ns), '6\n')
        self.assertNotIn('assert', split_checks(source)[0])

    def test_shipped_js_harness_executes_checks_even_during_prefix_replay(self):
        js_path = HERE.parent / 'practice' / 'notebook.js'
        # Run the actual JS array through Node, then execute that Python harness.
        probe = r'''
const fs = require('fs'), vm = require('vm');
let src = fs.readFileSync(process.argv[1], 'utf8');
src = src.replace('return { mount, runSource, splitChecks,', 'return { HARNESS, mount, runSource, splitChecks,');
const ctx = {window: {}}; vm.createContext(ctx); vm.runInContext(src, ctx);
process.stdout.write(ctx.window.LessonNotebook.HARNESS);
'''
        harness = subprocess.check_output(['node', '-e', probe, str(js_path)], text=True)
        ns = {}
        exec(harness, ns)
        code = 'value = 8\nvalue\n# Hidden checks\nassert _delta_output == "8\\n"'
        ns['_delta_cell'](code, '<test>', False)
        with self.assertRaises(AssertionError):
            ns['_delta_cell'](code.replace('value = 8', 'value = 9'), '<test>', False)
        # Every compiled teaching cell uses this same harness, not only the validator.
        import re
        count = 0
        for lesson in json.loads((HERE / 'lessons_structured.json').read_text())['lessons']:
            for kp in lesson['kps']:
                for segment in kp['segments']:
                    scope = {}
                    exec(harness, scope)
                    for field in ('concept_markdown', 'worked_example_markdown'):
                        for source in re.findall(r'^```python\s*\n(.*?)^```', segment[field], re.M | re.S):
                            scope['_delta_cell'](source, '<corpus>', False)
                            count += 1
        self.assertGreater(count, 250)


if __name__ == '__main__':
    unittest.main()
