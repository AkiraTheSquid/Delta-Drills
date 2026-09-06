"""watch.py — health checks for the pytorch-lesson KP sources (lesson tr-1)

Four pages, all in the PyTorch dialect, all fading into function-mode drills
q798–q841. The cheap structural checks worth running on every edit:

  * every page declares the tr-1 frontmatter contract (kc, supporting,
    new_syntax, faded/independent/integrated lists) and its kc is in the
    registry under lesson tr-1;
  * no page reintroduces a symbol ARENA never uses — `fill_` and
    `repeat_interleave` were removed on 2026-09-06 because the grounding
    ratchet rejects them, and a copy-paste from a numpy page could bring
    them back;
  * every faded fence has at least one blank, and every drill id the page
    claims exists in the bank.

Files are parsed, never executed — scripts/validate_lessons.py runs the
fences under torch and is the slow gate. Runs via `mod watch` — exit 0 =
PASS, exit non-zero = FAIL.
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

_DIR = os.path.dirname(os.path.abspath(__file__))
_LESSONS_DIR = os.path.normpath(os.path.join(_DIR, '..'))
_REGISTRY = os.path.join(_LESSONS_DIR, 'kc_registry.json')
_BANK = os.path.normpath(os.path.join(_LESSONS_DIR, '..', 'questions.json'))
_FORBIDDEN = ('.fill_(', 'repeat_interleave')


def _pages():
    return sorted(glob.glob(os.path.join(_DIR, 'kp-*.md')))


def _frontmatter(text):
    m = re.match(r'---\n(.*?)\n---\n', text, re.S)
    assert m, 'missing frontmatter'
    fm = {}
    for line in m.group(1).splitlines():
        k, _, v = line.partition(':')
        fm[k.strip()] = v.strip()
    return fm


def _ids(v):
    return [int(x) for x in re.findall(r'\d+', v)]


def check_imports():
    assert _pages(), 'no kp-*.md pages in lessons/pytorch'
    assert os.path.exists(_REGISTRY), 'kc_registry.json missing'


def check_public_api():
    reg = json.load(open(_REGISTRY, encoding='utf-8'))
    kcs = {k['id']: k for k in reg['kcs']}
    for path in _pages():
        fm = _frontmatter(open(path, encoding='utf-8').read())
        for key in ('kc', 'supporting', 'new_syntax', 'faded', 'independent', 'integrated'):
            assert key in fm, f'{os.path.basename(path)}: frontmatter lacks {key}'
        kc = kcs.get(fm['kc'])
        assert kc, f'{os.path.basename(path)}: kc {fm["kc"]} not in registry'
        assert kc['lesson'] == 'tr-1', f'{fm["kc"]} is not under lesson tr-1'


def check_invariants():
    bank_ids = set()
    if os.path.exists(_BANK):
        bank_ids = {q['id'] for q in json.load(open(_BANK, encoding='utf-8'))}
    for path in _pages():
        name = os.path.basename(path)
        text = open(path, encoding='utf-8').read()
        for bad in _FORBIDDEN:
            assert bad not in text, f'{name}: uses {bad!r}, which no ARENA notebook uses'
        for fence in re.findall(r'```python starter\n(.*?)```', text, re.S):
            assert '_____' in fence, f'{name}: a faded starter has no blank'
        fm = _frontmatter(text)
        claimed = _ids(fm['faded']) + _ids(fm['independent']) + _ids(fm['integrated'])
        assert len(_ids(fm['faded'])) >= 2, f'{name}: Faded floor is 2'
        assert len(_ids(fm['independent'])) >= 6, f'{name}: Solo floor is 6'
        assert len(_ids(fm['integrated'])) >= 3, f'{name}: Integrated floor is 3'
        if bank_ids:
            missing = [q for q in claimed if q not in bank_ids]
            assert not missing, f'{name}: ids not in questions.json: {missing}'


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
    print('PASS lessons/pytorch')
