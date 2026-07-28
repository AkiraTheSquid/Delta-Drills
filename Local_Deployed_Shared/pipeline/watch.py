"""watch.py — health checks for pipeline

This folder builds the served question bank and gates the deploy on it. The
defect these checks exist for is SILENT DIVERGENCE: the override layer list
and the override field whitelist are deliberately duplicated between the
exporter here and backend/app/questions.py (so the backend needs no pipeline
import), and nothing else notices when only one side is edited. The symptom
is a conversion or fix that "doesn't take" in one environment — see the
README's layer-list drift entry.

Sources are parsed, never imported: importing the exporter would rewrite the
bank artifacts as a side effect of a health check.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import ast
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_DIR, '..', '..'))
_EXPORTER = os.path.join(_DIR, 'export_questions_json.py')
_BACKEND_QUESTIONS = os.path.join(
    _REPO_ROOT, 'This-Directory-Only', 'backend', 'app', 'questions.py'
)

# Scripts that must stay syntactically loadable — the deploy runs these and a
# SyntaxError here fails the build after the bank has already been rewritten.
_GATE_SCRIPTS = (
    'export_questions_json.py',
    'audit_question_bank.py',
    'test_torch_grading.py',
    'validate_function_bank.py',
    'mech_gate_candidate.py',
)


def _source(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


def _module_jsonl_constants(tree):
    """Module-level names bound to a .jsonl filename.

    The exporter names its base layer through a constant
    (FUNCTION_OVERRIDES_PATH = CHATGPT_RUNTIME_DIR / "function_mode_overrides.jsonl")
    while the backend passes the same filename as a literal. Both load it
    first; only the spelling differs, so the constant has to be resolved or
    the two lists compare unequal for no real reason.
    """
    consts = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        literals = [
            c.value for c in ast.walk(node.value)
            if isinstance(c, ast.Constant)
            and isinstance(c.value, str)
            and c.value.endswith('.jsonl')
        ]
        if names and literals:
            consts[names[0]] = literals[0]
    return consts


def _override_layers(path, func_name):
    """The .jsonl layer filenames, in merge order, from a loader function.

    Ordered by source position — merge order is last-wins, so a set
    comparison would miss a genuine reordering.
    """
    src = _source(path)
    tree = ast.parse(src)
    consts = _module_jsonl_constants(tree)

    func = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func_name),
        None,
    )
    assert func is not None, f'{os.path.basename(path)} has no {func_name}()'

    found = []
    for node in ast.walk(func):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value.endswith('.jsonl'):
            found.append((node.lineno, node.col_offset, node.value))
        elif isinstance(node, ast.Name) and node.id in consts:
            found.append((node.lineno, node.col_offset, consts[node.id]))
    return [value for _, _, value in sorted(found)]


def _override_fields(path):
    """Field names an override may replace: override.get("x") / "x" in override."""
    src = _source(path)
    return set(re.findall(r'override\.get\(\s*"([^"]+)"', src)) | set(
        re.findall(r'"([^"]+)"\s+in\s+override', src)
    )


# ── Import checks ──────────────────────────────
def check_imports():
    """Every deploy-gating script must at least parse."""
    for name in _GATE_SCRIPTS:
        path = os.path.join(_DIR, name)
        assert os.path.exists(path), f'missing gate script: {name}'
        ast.parse(_source(path), filename=name)


# ── Public API checks ─────────────────────────
def check_public_api():
    """The two duplicated loaders both exist and both merge at least one layer."""
    exporter_layers = _override_layers(_EXPORTER, 'load_function_overrides')
    assert exporter_layers, 'exporter load_function_overrides() merges no layers'

    assert os.path.exists(_BACKEND_QUESTIONS), (
        f'backend questions.py not found at {_BACKEND_QUESTIONS}'
    )
    backend_layers = _override_layers(_BACKEND_QUESTIONS, '_load_function_overrides')
    assert backend_layers, 'backend _load_function_overrides() merges no layers'


# ── Invariant checks ──────────────────────────
def check_invariants():
    exporter_layers = _override_layers(_EXPORTER, 'load_function_overrides')
    backend_layers = _override_layers(_BACKEND_QUESTIONS, '_load_function_overrides')

    # Order matters: layers merge last-wins, so the same set in a different
    # sequence still resolves conflicting ids differently.
    assert exporter_layers == backend_layers, (
        'override layer lists diverged — exporter and backend would build '
        'different banks.\n'
        f'  exporter only: {[x for x in exporter_layers if x not in backend_layers]}\n'
        f'  backend only:  {[x for x in backend_layers if x not in exporter_layers]}\n'
        f'  exporter order: {exporter_layers}\n'
        f'  backend order:  {backend_layers}'
    )

    # Every registered layer must be a real file, or its edits vanish with no
    # error: _read_jsonl_overrides returns {} for a missing path. chatgpt/ is
    # gitignored, so a layer can also exist locally and never reach deploy.
    chatgpt_dir = os.path.join(_REPO_ROOT, 'This-Directory-Only', 'chatgpt')
    missing = [n for n in exporter_layers if not os.path.exists(os.path.join(chatgpt_dir, n))]
    assert not missing, f'registered override layers do not exist: {missing}'

    # A field absent from one whitelist is silently ignored there rather than
    # erroring — the override appears to apply everywhere except one artifact.
    exporter_fields = _override_fields(_EXPORTER)
    backend_fields = _override_fields(_BACKEND_QUESTIONS)
    assert exporter_fields == backend_fields, (
        'override field whitelists diverged — an override field would apply in '
        'one place and be silently dropped in the other.\n'
        f'  exporter only: {sorted(exporter_fields - backend_fields)}\n'
        f'  backend only:  {sorted(backend_fields - exporter_fields)}'
    )


# ── Run all checks ────────────────────────────
if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
