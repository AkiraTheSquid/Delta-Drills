"""watch.py — health checks for arena-book

This folder is a Jupyter Book build config, not a Python package. There is no
public Python API to call. Checks instead verify config integrity, symlink
resolution, and phase-1 invariants (Colab-only — no inline execution).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _read_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _yaml_load(path):
    """Parse YAML without requiring PyYAML. Falls back to a tolerant text scan
    when PyYAML isn't on sys.path (mod watch runs in a minimal env).
    """
    try:
        import yaml  # type: ignore
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except ImportError:
        return None


def check_imports():
    # No Python modules in this folder. Nothing to import.
    return


def check_public_api():
    # Static-site build config — no callable API surface.
    return


def check_invariants():
    # 1. Required files exist.
    for name in ('_config.yml', '_toc.yml', 'intro.md', 'requirements.txt', 'README.md'):
        path = os.path.join(HERE, name)
        assert os.path.isfile(path), f"missing required file: {name}"

    # 2. Chapter symlinks resolve to real ARENA dirs.
    expected_chapters = (
        'chapter0_fundamentals',
        'chapter1_transformer_interp',
        'chapter2_rl',
        'chapter3_llm_evals',
    )
    for chapter in expected_chapters:
        path = os.path.join(HERE, chapter)
        assert os.path.islink(path), f"{chapter} should be a symlink into the ARENA tree"
        assert os.path.isdir(path), f"{chapter} symlink does not resolve — ARENA tree moved?"

    # 3. README marker removed (filled in).
    readme = _read_text(os.path.join(HERE, 'README.md'))
    assert 'modulario:template' not in readme, "README.md still has the modulario template marker"

    # 4. Phase-1 invariants in _config.yml — Colab only, no inline execution.
    config_path = os.path.join(HERE, '_config.yml')
    parsed = _yaml_load(config_path)
    if parsed is not None:
        launch = (parsed.get('launch_buttons') or {})
        thebe = launch.get('thebe', False)
        assert thebe is False, (
            "launch_buttons.thebe must remain false — phase 1 is Colab-only. "
            "If turning on inline execution, see "
            "~/.claude/projects/-home-stellar-thread/memory/delta-drills-jupyter-book-execution-decision.md"
        )
        execute = (parsed.get('execute') or {})
        mode = execute.get('execute_notebooks', 'off')
        assert mode in ('off', False), (
            "execute.execute_notebooks must be 'off' — build env has no PyTorch/GPU"
        )
    else:
        # PyYAML not available — fall back to text scan.
        config_text = _read_text(config_path)
        assert 'thebe: false' in config_text, "expected `thebe: false` in _config.yml (phase-1 guard)"
        assert 'execute_notebooks: "off"' in config_text or "execute_notebooks: 'off'" in config_text, (
            "expected execute_notebooks: \"off\" in _config.yml"
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
