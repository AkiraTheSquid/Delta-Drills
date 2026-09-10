"""Execute authored lesson cells with the same output/check contract as the UI."""
import ast
import contextlib
import io
import re

MARKER = '\n# Hidden checks\n'


def split_checks(source):
    visible, marker, hidden = source.partition(MARKER)
    return visible, hidden if marker else ''


def _assertions(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            if isinstance(node.test, ast.Constant):
                raise AssertionError('An assertion must check a result, not a constant')
            if isinstance(node.test, ast.Compare) and len(node.test.comparators) == 1:
                if ast.dump(node.test.left) == ast.dump(node.test.comparators[0]):
                    raise AssertionError('An assertion must not compare a value with itself')
            yield node


def run_checked(source, namespace=None, *, require_assert=True):
    """Echo the last expression, capture actual output, then run hidden checks.

    Hidden checks get a shallow namespace copy so helper names cannot overwrite
    the next example's names. Their assertions still inspect the actual values.
    Count assertions reached at runtime: an uncalled function or dead branch
    containing `assert` is not a check of this cell.
    """
    if not __debug__:
        raise AssertionError('Lesson checks require assertions enabled')
    ns = namespace if namespace is not None else {}
    visible, hidden = split_checks(source)
    trees = [ast.parse(visible, '<lesson>'), ast.parse(hidden, '<lesson checks>')]
    sites = [(tree, list(_assertions(tree))) for tree in trees]
    if require_assert and not any(nodes for _, nodes in sites):
        raise AssertionError('Runnable lesson cell has no assertion')
    reached = []

    class CountAssertions(ast.NodeTransformer):
        def visit_Assert(self, node):
            call = ast.Expr(ast.Call(ast.Name('_delta_reached', ast.Load()), [], []))
            return [ast.copy_location(call, node), node]

    trees = [ast.fix_missing_locations(CountAssertions().visit(tree)) for tree in trees]
    ns['_delta_reached'] = lambda: reached.append(True)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        tree = trees[0]
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            head = ast.Module(body=tree.body[:-1], type_ignores=[])
            exec(compile(head, '<lesson>', 'exec'), ns)
            value = eval(compile(ast.Expression(tree.body[-1].value), '<lesson>', 'eval'), ns)
            if value is not None:
                print(repr(value))
        else:
            exec(compile(tree, '<lesson>', 'exec'), ns)
    check_ns = dict(ns, _delta_output=output.getvalue())
    exec(compile(trees[1], '<lesson checks>', 'exec'), check_ns)
    if require_assert and not reached:
        raise AssertionError('No assertion executed; a dormant assert does not check this cell')
    return output.getvalue()


def check_compiled_examples(directory):
    """All runnable teaching cells, fresh namespace for each concept page."""
    import json
    from pathlib import Path
    directory = Path(directory)
    data = json.loads((directory / 'lessons_structured.json').read_text())
    authored = {}
    for path in directory.glob('*/kp-*.md'):
        text = path.read_text()
        kc = re.search(r'^kc: (.+)$', text, re.M)
        if kc:
            authored[kc[1]] = re.findall(r'^```python\s*\n(.*?)^```', text, re.M | re.S)
    count = 0
    for lesson in data['lessons']:
        for kp in lesson['kps']:
            compiled = [source for segment in kp['segments']
                        for field in ('concept_markdown', 'worked_example_markdown')
                        for source in re.findall(r'^```python\s*\n(.*?)^```', segment[field], re.M | re.S)]
            assert authored.get(kp['kc']) == compiled, (
                f"{kp['kc']}: teaching source changed; validate and recompile lessons")
            for segment in kp['segments']:
                ns = {}
                for field in ('concept_markdown', 'worked_example_markdown'):
                    for source in re.findall(r'^```python\s*\n(.*?)^```', segment[field], re.M | re.S):
                        try:
                            run_checked(source, ns)
                        except Exception as exc:
                            raise AssertionError(f"{kp['kc']}/{segment['concept_id']}/{field}: {exc}") from exc
                        count += 1
            for field in ('guided_items', 'applied_items', 'solo_items'):
                for item in kp.get(field, []):
                    source = item.get('worked_example_code')
                    if source:
                        try:
                            run_checked(source)
                        except Exception as exc:
                            raise AssertionError(f"{kp['kc']}/q{item['question_id']}: {exc}") from exc
                        count += 1
    return count


if __name__ == '__main__':
    from pathlib import Path
    print(f'PASS: {check_compiled_examples(Path(__file__).parent)} runnable cells asserted')
