"""watch.py — health checks for the einsum-lesson KP sources

These 10 markdown files are the taught side of es-1..es-2; the drills they
fade into live in the question bank. The defect worth catching cheaply is
DIALECT DRIFT between the two halves.

The bank is being converted from NumPy to the PyTorch dialect ARENA uses, one
lesson at a time. While that is in progress a KP file and the questions it
embeds can disagree — the page teaches `np.einsum` while the
drill grades `t.einsum`. Nothing else notices: the exercise still passes its own
tests, and the learner just gets whiplash.

So: every KP is required to be internally consistent, and to match the dialect
of the bank questions it fades into. Files are parsed, never executed —
scripts/validate_lessons.py already runs the fences (with torch, so it needs
the backend venv) and it is the slow, thorough gate. This is the fast
structural one.

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
_EINSUM_USE = re.compile(r'\beinsum\s*\(')


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


def check_imports():
    """The artifacts these files are validated against must be loadable."""
    paths = _kp_paths()
    assert paths, 'no kp-*.md files found — the einsum lesson content is missing'
    for path in paths:
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
        assert _frontmatter(text).get('kc'), f'{os.path.basename(path)}: no kc in frontmatter'

    for name in ('kc_registry.json', 'qmatrix_tags.json'):
        with open(os.path.join(_LESSONS_DIR, name), encoding='utf-8') as fh:
            json.load(fh)


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

        fences = _fence_bodies(text)

        # One dialect per page. A file that imports both is mid-conversion.
        dialects = {d for d in (_dialect(c) for c in fences) if d}
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

        # Folder-specific: these pages exist to teach einsum, so each one
        # has to actually show it. A page whose worked example drifted to `@`
        # or `.sum(dim=)` still validates and still reads fine — it just no
        # longer teaches the notation its KC claims.
        assert any(_EINSUM_USE.search(code) for code in fences), (
            f'{name}: no fence calls einsum — the page no longer demonstrates '
            'the notation it is named for'
        )


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
