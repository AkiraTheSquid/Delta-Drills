"""watch.py — health checks for the numpy-lesson KP sources

These 44 markdown files are the taught side of the np-1..np-4 lessons; the
drills they fade into live in the question bank. The defect worth catching
cheaply is DIALECT DRIFT between the two halves.

The bank is being converted from NumPy to the PyTorch dialect ARENA uses, one
lesson at a time (np-1 first). While that is in progress a KP file and the
questions it embeds can disagree — the page teaches `np.repeat` while the
drill grades `t.repeat_interleave`. Nothing else notices: the exercise still
passes its own tests, and the learner just gets whiplash.

So: every KP is required to be internally consistent, and to match the
dialect of the bank questions it fades into. Files are parsed, never
executed — scripts/validate_lessons.py already runs the fences (with torch,
so it needs the backend venv) and it is the slow, thorough gate. This is the
fast structural one.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import glob
import json
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

_DIR = os.path.dirname(os.path.abspath(__file__))
_LESSONS_DIR = os.path.normpath(os.path.join(_DIR, '..'))
_BANK = os.path.normpath(os.path.join(_LESSONS_DIR, '..', 'questions.json'))

# A dialect marker is an import, not a bare mention: prose may name the other
# library while explaining the difference, and that is exactly what a good
# conversion does.
_TORCH_IMPORT = re.compile(r'(?m)^\s*(?:import\s+torch\b|from\s+torch[\s.])')
_NUMPY_IMPORT = re.compile(r'(?m)^\s*(?:import\s+numpy\b|from\s+numpy[\s.])')


def _kp_paths():
    return sorted(glob.glob(os.path.join(_DIR, 'kp-*.md')))


def _frontmatter(text):
    """The YAML-ish header block. Only the flat scalars and [a, b] lists that
    these files actually use are supported — no YAML dependency."""
    if not text.startswith('---'):
        return {}
    end = text.find('\n---', 3)
    if end == -1:
        return {}
    out = {}
    for line in text[3:end].splitlines():
        if not line.strip() or ':' not in line:
            continue
        key, _, value = line.partition(':')
        value = value.strip()
        if value.startswith('[') and value.endswith(']'):
            inner = value[1:-1].strip()
            out[key.strip()] = [v.strip() for v in inner.split(',')] if inner else []
        else:
            out[key.strip()] = value
    return out


def _fence_bodies(text):
    """Code inside ```python fences, including the starter/solution variants."""
    return [m.group(1) for m in re.finditer(r'```python[^\n]*\n(.*?)```', text, re.S)]


def _dialect(code):
    torch, numpy = bool(_TORCH_IMPORT.search(code)), bool(_NUMPY_IMPORT.search(code))
    if torch and not numpy:
        return 'torch'
    if numpy and not torch:
        return 'numpy'
    return 'mixed' if (torch and numpy) else None


# ── Import checks ──────────────────────────────
def check_imports():
    """The artifacts these files are validated against must be loadable."""
    paths = _kp_paths()
    assert paths, 'no kp-*.md files found — the numpy lesson content is missing'
    for path in paths:
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
        assert _frontmatter(text).get('kc'), f'{os.path.basename(path)}: no kc in frontmatter'

    for name in ('kc_registry.json', 'qmatrix_tags.json'):
        with open(os.path.join(_LESSONS_DIR, name), encoding='utf-8') as fh:
            json.load(fh)


# ── Public API checks ─────────────────────────
def check_public_api():
    """Shape the lesson compiler and the in-app player rely on: every KP names
    a KC that exists, and carries at least one exercise to fade into."""
    with open(os.path.join(_LESSONS_DIR, 'kc_registry.json'), encoding='utf-8') as fh:
        known_kcs = {kc['id'] for kc in json.load(fh)['kcs']}

    for path in _kp_paths():
        name = os.path.basename(path)
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
        meta = _frontmatter(text)

        assert meta['kc'] in known_kcs, f'{name}: kc {meta["kc"]!r} is not in kc_registry.json'
        exercises = (meta.get('faded') or []) + (meta.get('guided') or []) \
            + (meta.get('independent') or [])
        assert exercises, f'{name}: no faded/guided/independent question ids'
        assert '## Concept' in text, f'{name}: no Concept section'


# ── Invariant checks ──────────────────────────
def check_invariants():
    with open(_BANK, encoding='utf-8') as fh:
        bank = {q['id']: q for q in json.load(fh)}

    for path in _kp_paths():
        name = os.path.basename(path)
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
        meta = _frontmatter(text)

        # Every referenced drill must still be in the bank, or the lesson
        # fades into a question the learner can never be served.
        referenced = [int(q) for key in ('faded', 'guided', 'independent')
                      for q in (meta.get(key) or [])]
        missing = [q for q in referenced if q not in bank]
        assert not missing, f'{name}: references question ids not in the bank: {missing}'

        # One dialect per page. A file that imports both is mid-conversion.
        dialects = {d for d in (_dialect(c) for c in _fence_bodies(text)) if d}
        assert dialects != {'mixed'} and 'mixed' not in dialects, \
            f'{name}: a single code fence imports both numpy and torch'
        assert len(dialects) <= 1, \
            f'{name}: mixes dialects across fences: {sorted(dialects)}'
        # An undeclared page would slip past the drill comparison below, so it
        # is a failure rather than a skip. Every KP in this course opens its
        # fences with the import; a page without one is unfinished.
        assert dialects, (
            f'{name}: no fence declares a dialect — add the import so the '
            'lesson can be checked against its drills'
        )
        page = dialects.pop()

        # And it must be the dialect of the drills it fades into — this is the
        # whiplash check: teaching numpy and grading torch (or the reverse).
        for qid in referenced:
            drill = _dialect((bank[qid].get('answer_code') or '')
                             + '\n' + (bank[qid].get('starter_code') or ''))
            assert drill is None or drill == page, (
                f'{name}: page teaches {page} but q{qid} is a {drill} drill — '
                'convert the lesson and its questions in the same pass'
            )



# ── The three standing content guards ─────────
# Filled 2026-08-29 on Seth's instruction: these must fire on the folder being
# EDITED, not only from scripts/, so that adding a drill or a KP page here is
# what trips them. The implementations live in scripts/guard_checks.py — one
# copy, because a guard duplicated six times becomes six different guards
# inside a month. Scoped to numpy. so this watcher reports its own concepts.
import importlib.util as _ilu
_GUARD = os.path.join(os.path.join(_DIR, '..', '..', '..'), 'scripts', 'guard_checks.py')
_spec = _ilu.spec_from_file_location('guard_checks', _GUARD)
_guard = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_guard)

_CONTENT_GUARDS = _guard.run('numpy.')

if __name__ == '__main__':
    checks = [check_imports, check_public_api,
              check_invariants] + _CONTENT_GUARDS
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
