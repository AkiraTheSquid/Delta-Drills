"""watch.py — health checks for scripts/

Most files here are operator entry points (bash + thin Python runpy
wrappers). We deliberately do NOT import the wrappers — they execute
real pipeline code as a side effect of `runpy.run_path`. Instead we
verify the wrapper pattern by reading the file text, and fully import
only the pure-logic helpers (refresh_split_layout).
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SHARED_PIPELINE = os.path.join(REPO_ROOT, 'Local_Deployed_Shared', 'pipeline')

PIPELINE_WRAPPERS = (
    'build_function_bank.py',
    'build_function_mode_repair_requests.py',
    'build_function_mode_requests.py',
    'export_questions_json.py',
    'test_function_validator.py',
    'validate_function_bank.py',
)

LONG_RUNNING_PY_SCRIPTS = (
    'watch_delta_drills_dev.py',
)

SHELL_SCRIPTS = (
    'deploy_delta_drills.sh',
    'deploy_delta_drills_colab.sh',
    'deploy_delta_drills_local.sh',
    'sync-deploy.sh',
    'sync-local.sh',
    'build_arena_book.sh',
)


def _read(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _import_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_imports():
    # Only refresh_split_layout is safe to import — pure stdlib, has __main__ guard.
    path = os.path.join(HERE, 'refresh_split_layout.py')
    assert os.path.isfile(path), 'refresh_split_layout.py missing'
    module = _import_module_from_path('refresh_split_layout', path)
    assert hasattr(module, 'main'), 'refresh_split_layout.main missing'
    assert hasattr(module, 'ALLOWED_ROOT_NAMES'), 'ALLOWED_ROOT_NAMES constant missing'


def check_public_api():
    # Shell scripts: shebang + executable bit.
    for name in SHELL_SCRIPTS:
        path = os.path.join(HERE, name)
        assert os.path.isfile(path), f'shell script missing: {name}'
        first = _read(path).splitlines()[0] if os.path.getsize(path) else ''
        assert first.startswith('#!'), f'{name} missing shebang'
        assert os.access(path, os.X_OK), f'{name} is not executable'

    # Python wrappers: thin runpy.run_path stubs that delegate to pipeline/.
    for name in PIPELINE_WRAPPERS:
        path = os.path.join(HERE, name)
        assert os.path.isfile(path), f'wrapper missing: {name}'
        text = _read(path)
        assert 'runpy.run_path' in text, f'{name} is not a runpy wrapper'
        target = os.path.join(SHARED_PIPELINE, name)
        assert os.path.isfile(target), (
            f'{name} delegates to Local_Deployed_Shared/pipeline/{name} which does not exist'
        )

    for name in LONG_RUNNING_PY_SCRIPTS:
        path = os.path.join(HERE, name)
        assert os.path.isfile(path), f'python script missing: {name}'
        text = _read(path)
        assert 'watching Chrome debug port' in text, f'{name} missing isolation watcher docstring'


def check_invariants():
    # 1. refresh_split_layout's allowlist still covers the split layout.
    module = _import_module_from_path(
        'refresh_split_layout',
        os.path.join(HERE, 'refresh_split_layout.py'),
    )
    required = {'Local_Deployed_Shared', 'This-Directory-Only', '.git'}
    missing = required - set(module.ALLOWED_ROOT_NAMES)
    assert not missing, f'refresh_split_layout.ALLOWED_ROOT_NAMES missing: {missing}'

    # 2. Deploy script still calls the arena-book build helper.
    deploy_text = _read(os.path.join(HERE, 'deploy_delta_drills.sh'))
    assert 'build_arena_book.sh' in deploy_text, (
        'deploy_delta_drills.sh no longer references build_arena_book.sh — '
        'arena-book deploys will silently skip the rebuild.'
    )

    # 2b. The main deploy still republishes the Colab edition.
    # delta-drills-colab.vercel.app is the same frontend under a second
    # project; when only the main deploy runs, the fork keeps serving the
    # previous build and looks like a broken app rather than a stale one.
    assert 'deploy_delta_drills_colab.sh' in deploy_text, (
        'deploy_delta_drills.sh no longer runs deploy_delta_drills_colab.sh — '
        'the Colab edition will silently drift behind the main deploy.'
    )

    # 3. Build script writes to the expected staging dir under Local_Deployed_Shared.
    build_text = _read(os.path.join(HERE, 'build_arena_book.sh'))
    assert 'Local_Deployed_Shared/arena-book' in build_text, (
        'build_arena_book.sh staging path drift — arena-book output may not '
        'land where vercel.json expects.'
    )

    # 4. extract_arena_prereqs points at a notebook path that actually resolves.
    extract_text = _read(os.path.join(HERE, 'extract_arena_prereqs.py'))
    assert 'ARENA_3.0-main' in extract_text or 'ARENA_4.0-main' in extract_text, (
        'extract_arena_prereqs.py: ARENA snapshot dir reference missing'
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
