"""watch.py — health checks for lessons

This folder is the SOURCE OF TRUTH for what the tutor may teach: the ITS serves
only questions carrying a `target_kcs` entry in qmatrix_tags.json (see
docs/decision-kc-only-serving.md). That makes a silent corruption here
equivalent to silently un-teaching part of the curriculum, so these checks
guard the structure rather than the prose.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import json
import sys
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, '../..'))

# Chapter 1 is authored and frozen as the validation reference set. A drop here
# means KCs were lost, not that new ones were added — growth is expected.
# Lowered from 64 to 63 on 2026-07-28: numpy.structured-dtypes was RETIRED, not
# lost. Record dtypes, datetime64 and genfromtxt have no PyTorch form — a tensor
# is homogeneous — so the KC could not follow the course into torch.
_MIN_KCS = 63
# 380 -> 370 on 2026-07-28. The retirements cost exactly 7 tagged questions
# (six numpy.structured-dtypes drills plus q65), leaving 373. An earlier edit
# in the same pass dropped this to 343, which was 30 questions of unexplained
# slack -- a floor that loose would let a quarter of a lesson fall out of the
# ITS without failing. 370 keeps the small headroom a floor needs and nothing
# more.
_MIN_TAGGED_QUESTIONS = 370


def _read(name):
    with open(os.path.join(_DIR, name), encoding='utf-8') as fh:
        return json.load(fh)


def check_imports():
    """The three artifacts the backend and lesson gate load must parse."""
    for name in ('kc_registry.json', 'qmatrix_tags.json', 'lessons_structured.json'):
        _read(name)


def check_public_api():
    """Shape the consumers rely on: backend/app/lessons.py reads kcs[].id and
    target_kcs; export_kc_difficulty.py and export_kc_atom_crosswalk.py read the
    same two files."""
    registry = _read('kc_registry.json')
    kcs = registry.get('kcs')
    assert isinstance(kcs, list) and kcs, 'kc_registry.json has no kcs[]'
    for kc in kcs:
        assert kc.get('id'), f'KC without an id: {kc!r}'
    assert isinstance(registry.get('lessons'), list), 'kc_registry.json has no lessons[]'

    qmatrix = _read('qmatrix_tags.json')
    assert isinstance(qmatrix, dict) and qmatrix, 'qmatrix_tags.json is not a non-empty object'


def check_invariants():
    registry = _read('kc_registry.json')
    qmatrix = _read('qmatrix_tags.json')
    ids = {kc['id'] for kc in registry['kcs']}

    assert len(ids) == len(registry['kcs']), 'duplicate KC ids in kc_registry.json'
    assert len(ids) >= _MIN_KCS, f'KC count fell to {len(ids)}, expected >= {_MIN_KCS}'

    # Prerequisite lattice must stay a resolvable DAG — the graph draws it and
    # the estimator's structural channel walks it.
    prereqs = {kc['id']: (kc.get('prereqs') or []) for kc in registry['kcs']}
    dangling = {p for ps in prereqs.values() for p in ps if p not in ids}
    assert not dangling, f'prereqs point at unknown KCs: {sorted(dangling)}'

    state = {}  # 0 = visiting, 1 = done

    def visit(node):
        if state.get(node) == 1:
            return
        assert state.get(node) != 0, f'prerequisite cycle through {node}'
        state[node] = 0
        for parent in prereqs.get(node, []):
            visit(parent)
        state[node] = 1

    for kc in ids:
        visit(kc)

    # Every target KC must exist, or its questions become unservable silently.
    # Count only tags on question ids that are REALLY in the bank: a q-matrix
    # full of valid KC names under stale ids would otherwise satisfy the
    # coverage floor while parking the entire selection pool.
    with open(os.path.join(_DIR, '..', 'questions.json'), encoding='utf-8') as fh:
        bank_ids = {str(q['id']) for q in json.load(fh)}

    tagged = 0
    unknown = set()
    orphan_qids = set()
    for qid, tags in qmatrix.items():
        targets = tags.get('target_kcs') or []
        unknown.update(kc for kc in targets if kc not in ids)
        if not targets:
            continue
        if str(qid) in bank_ids:
            tagged += 1
        else:
            orphan_qids.add(str(qid))

    assert not unknown, f'qmatrix target_kcs reference unknown KCs: {sorted(unknown)}'
    assert not orphan_qids, (
        f'qmatrix tags {len(orphan_qids)} question ids absent from questions.json: '
        f'{sorted(orphan_qids)[:10]}'
    )
    assert tagged >= _MIN_TAGGED_QUESTIONS, (
        f'only {tagged} bank questions carry target_kcs, expected >= {_MIN_TAGGED_QUESTIONS} — '
        'questions silently parked out of the ITS'
    )



# ── The two standing content guards ───────────
# Filled 2026-08-29 on Seth's instruction: these must fire on the folder being
# EDITED, not only from scripts/, so that adding a drill or a KP page here is
# what trips them. The implementations live in scripts/guard_checks.py — one
# copy, because a guard duplicated six times becomes six different guards
# inside a month. Scoped to the whole bank so this watcher reports its own concepts.
import importlib.util as _ilu
_GUARD = os.path.join(os.path.join(_DIR, '..', '..'), 'scripts', 'guard_checks.py')
_spec = _ilu.spec_from_file_location('guard_checks', _GUARD)
_guard = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_guard)

_CONTENT_GUARDS = _guard.run(None)

if __name__ == '__main__':
    checks = [check_imports, check_public_api,
              check_invariants] + _CONTENT_GUARDS
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
