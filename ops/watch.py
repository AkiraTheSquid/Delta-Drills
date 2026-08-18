"""watch.py — health checks for ops

This tree is operator tooling and is NOT deployed. The checks here pin the one
rule that makes that safe — the dependency only ever points inward — because
breaking it fails in production rather than here.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import os
import re
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(THIS)

sys.path.insert(0, REPO)


def _python_files(root):
    for base, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.venv', 'node_modules', '.git')]
        for name in names:
            if name.endswith('.py'):
                yield os.path.join(base, name)


def check_imports():
    """Every script here parses, and each subfolder carries its own runner."""
    import ast

    for path in _python_files(THIS):
        with open(path, encoding='utf-8') as fh:
            try:
                ast.parse(fh.read())
            except SyntaxError as exc:
                raise AssertionError(f"{os.path.relpath(path, REPO)} does not parse: {exc}")

    for entry in sorted(os.listdir(THIS)):
        folder = os.path.join(THIS, entry)
        if os.path.isdir(folder) and not entry.startswith(('.', '__')):
            assert os.path.exists(os.path.join(folder, 'README.md')), (
                f"ops/{entry}/ has no README — an operator script nobody can "
                "read is a script nobody will run"
            )


def check_public_api():
    """The runner must keep re-execing into the backend venv.

    Without it the script runs on system python, which has no torch, and the
    verification step that stops a broken reference answer reaching the bank
    silently reports "unavailable" instead of "wrong".
    """
    runner = os.path.join(THIS, 'question_repair', 'run_repairs.py')
    if not os.path.exists(runner):
        return
    with open(runner, encoding='utf-8') as fh:
        source = fh.read()
    assert 'def ensure_backend_python' in source, (
        "run_repairs.py no longer re-execs into the backend venv"
    )
    assert 'sys.prefix' in source, (
        "the venv check is back to comparing interpreter paths — .venv/bin/python "
        "is a SYMLINK to the system interpreter, so that comparison always "
        "matches and the re-exec never happens"
    )


def check_invariants():
    """Nothing the deployed app ships may import this tree.

    `ops/` is not in the image. An import from backend or shared code is an
    ImportError in production and nowhere else, which is the worst place to
    find out.
    """
    importers = []
    pattern = re.compile(r'(?m)^\s*(?:from|import)\s+ops\b')
    for area in ('This-Directory-Only/backend/app', 'Local_Deployed_Shared'):
        root = os.path.join(REPO, area)
        if not os.path.isdir(root):
            continue
        for path in _python_files(root):
            with open(path, encoding='utf-8', errors='replace') as fh:
                if pattern.search(fh.read()):
                    importers.append(os.path.relpath(path, REPO))
    assert not importers, (
        "deployed code imports ops/, which is not in the Docker image — "
        f"production ImportError in: {', '.join(importers)}"
    )

    # The other direction is the whole point of the folder: no model credential
    # lives here, because the model runs under Seth's own CLI login.
    leaks = []
    for path in _python_files(THIS):
        if os.path.abspath(path) == os.path.abspath(__file__):
            continue  # this file names the credentials in order to forbid them
        with open(path, encoding='utf-8', errors='replace') as fh:
            body = fh.read()
        for needle in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'sk-ant-'):
            if needle in body:
                leaks.append(f"{os.path.relpath(path, REPO)}:{needle}")
    assert not leaks, (
        "an ops script reaches for a model API key — these run through the "
        f"local `claude` CLI on purpose: {', '.join(leaks)}"
    )


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
