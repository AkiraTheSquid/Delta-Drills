"""watch.py — health checks for visual_review

Local-only review app for image-output questions (server.py). These checks
guard the contract the rest of the repo relies on: the server still reads the
full bank, its review-state files stay valid JSON/JSONL, and every question
id recorded in a review decision still exists in the bank — a stale id is
reported so a "Needs check" flag never points at nothing unnoticed.
Filled 2026-09-06. Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


import ast
import json

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
BANK = os.path.join(REPO, 'This-Directory-Only', 'questions_full.json')

# Parsed rather than imported: server.py pulls in numpy/PIL and starts an HTTP
# server at module level, neither of which a watcher should need.
_REQUIRED_FUNCS = ('evaluate_question', 'load_questions', 'render_manifest',
                   'load_state', 'save_state', 'write_flags_exports')


def _tree():
    with open(os.path.join(HERE, 'server.py'), encoding='utf-8') as fh:
        return ast.parse(fh.read())


def check_imports():
    tree = _tree()  # SyntaxError here = the review server cannot start
    # The QUESTIONS_PATH assignment itself, not any mention of the filename:
    # a docstring or dead constant must not satisfy the full-bank contract.
    assigned = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'QUESTIONS_PATH' for t in node.targets):
            assigned = ast.unparse(node.value)
    assert assigned and 'questions_full.json' in assigned, (
        f'server.py QUESTIONS_PATH = {assigned!r} — it must review the FULL bank '
        '(This-Directory-Only/questions_full.json, solutions included), not the student export')
    loads = [ast.unparse(n) for n in ast.walk(tree)
             if isinstance(n, ast.Attribute) and n.attr == 'read_text'
             and 'QUESTIONS_PATH' in ast.unparse(n)]
    assert loads, 'server.py load_questions() no longer reads QUESTIONS_PATH'
    assert os.path.exists(BANK), f'{BANK} missing — run export_questions_json.py'


def check_public_api():
    names = {n.name for n in ast.walk(_tree()) if isinstance(n, ast.FunctionDef)}
    missing = [f for f in _REQUIRED_FUNCS if f not in names]
    assert not missing, f'server.py lost functions: {missing}'


def check_invariants():
    with open(BANK, encoding='utf-8') as fh:
        bank_ids = {str(q.get('id')) for q in json.load(fh)}
    state_path = os.path.join(HERE, 'review_state.json')
    if os.path.exists(state_path):
        with open(state_path, encoding='utf-8') as fh:
            state = json.load(fh)
        assert isinstance(state, dict), 'review_state.json must be an object'
        stale = sorted(k for k in _state_ids(state) if k not in bank_ids)
        # Reported, not fatal: review_state.json is a local audit artifact and
        # ids leave the bank legitimately (pipeline/retired_question_ids.json,
        # validator exclusions). 386/387/393 were already stale on 2026-09-06.
        if stale:
            print(f'WARN visual_review: {len(stale)} reviewed id(s) no longer in the '
                  f'bank: {stale[:10]} — prune or --regen', file=sys.stderr)
    flags = os.path.join(HERE, 'visual_malformed_flags.jsonl')
    if os.path.exists(flags):
        with open(flags, encoding='utf-8') as fh:
            for i, line in enumerate(fh, 1):
                if line.strip():
                    json.loads(line)  # one JSON object per line, or the log is unreadable


def _state_ids(state):
    """Question ids in review_state.json — the file keys decisions by id at the
    top level or under a 'reviews' map (current shape); accept either shape."""
    for key in ('reviews', 'questions', 'items'):
        if isinstance(state.get(key), dict):
            return [str(k) for k in state[key]]
    return [str(k) for k in state if str(k).isdigit()]


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
