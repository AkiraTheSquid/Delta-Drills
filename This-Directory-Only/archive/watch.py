"""watch.py — health checks for archive

This folder holds retired content, one dated subfolder per retirement. The
checks here are about the SHAPE of the archive; each dated subfolder's own
watch.py is what cross-checks its pages against the live registry.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

HERE = os.path.dirname(os.path.abspath(__file__))
DATED = re.compile(r'^[a-z0-9-]+-\d{4}-\d{2}-\d{2}$')


def _subfolders():
    return sorted(
        d for d in os.listdir(HERE)
        if os.path.isdir(os.path.join(HERE, d)) and not d.startswith('.')
        and d != '__pycache__'
    )


def check_imports():
    """Every retirement is a DATED folder — that is the whole filing system."""
    undated = [d for d in _subfolders() if not DATED.match(d)]
    assert not undated, (
        f'archive subfolders without a trailing date: {undated}. The date is what '
        'lets a reader tell which cut a page belongs to; an undated folder becomes '
        'a junk drawer within two retirements.'
    )


def check_public_api():
    """Each retirement carries its own reasoning and its own guard."""
    for d in _subfolders():
        for required in ('README.md', 'watch.py'):
            path = os.path.join(HERE, d, required)
            assert os.path.exists(path), f'{d}/ has no {required}'
        head = open(os.path.join(HERE, d, 'README.md'), encoding='utf-8').read(200)
        assert 'modulario:template' not in head, (
            f'{d}/README.md is still the unfilled template — a retirement with no '
            'stated reason cannot be reviewed or reversed'
        )


def check_invariants():
    """Nothing here may be reachable from the deployed tree."""
    shared = os.path.abspath(os.path.join(HERE, '..', '..', 'Local_Deployed_Shared'))
    assert not os.path.commonpath([HERE, shared]) == shared, (
        'the archive is inside Local_Deployed_Shared — retired content would be '
        'rsynced into the Deployed worktree and shipped'
    )


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
