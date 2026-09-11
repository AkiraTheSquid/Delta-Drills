import unittest

from content_safety import is_stub, leak_findings
from validate_lessons import grade_against_bank


class ContentSafetyTests(unittest.TestCase):
    def check_leak(self, prompt="Return pairwise scores.", starter="def solve(x):\n    pass\n", rung="integrated"):
        return leak_findings({"question_text": prompt, "starter_code": starter,
                              "answer_code": 'def solve(x):\n    return x.sum(dim=1)\n'}, rung, "test")

    def test_contract_is_not_a_recipe(self):
        self.assertFalse(self.check_leak("Return one total per row, shape (m,). x: float matrix (m,n)."))

    def test_integrated_prompt_code(self):
        self.assertTrue(self.check_leak("Return `x.sum(dim=1)`."))

    def test_comment_and_docstring_are_not_exempt(self):
        for comment in ('# x.sum(dim=1)', '"""Use x.sum(dim=1)."""'):
            self.assertTrue(self.check_leak(starter=f'def solve(x):\n    {comment}\n    pass\n'))

    def test_prose_recipe_without_code(self):
        self.assertTrue(self.check_leak("Build the edges first, then take their midpoints."))

    def test_partial_computation_in_starter(self):
        self.assertTrue(self.check_leak(starter='def solve(x):\n    y=x.sum(dim=1)\n    pass\n'))

    def test_pattern_string(self):
        self.assertTrue(leak_findings({"question_text": 'Use pattern "b c -> b".',
            "answer_code": 'def solve(x):\n    return einops.reduce(x,"b c -> b","sum")'}, "solo", "q"))

    def test_numpy_and_einops_calls(self):
        for lib in ('np', 'einops'):
            self.assertTrue(leak_findings({"question_text": f'Call {lib}.reshape(x).',
                "answer_code": f'def solve(x):\n    return {lib}.reshape(x)'}, "independent", "q"))

    def test_arena_stub_allowed_but_filled_body_not(self):
        self.assertTrue(is_stub('def solve(x):\n    raise NotImplementedError()'))
        self.assertFalse(is_stub('def solve(x):\n    return x+1'))

    def test_empty_test_set_cannot_validate(self):
        self.assertTrue(grade_against_bank('def solve(x): return 0', {"exercise": {"test_cases": []}}))


if __name__ == '__main__':
    unittest.main()
