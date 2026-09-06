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
# 63 -> 37 on 2026-08-30, the cut back to the ARENA-necessary path. 35 KCs were
# retired in one pass: the whole einsum course (ARENA writes einops.einsum in 61
# of its 458 notebooks and torch.einsum in NONE), all of np-4, and the np-2/np-3
# concepts whose own operations appear in fewer than ~5% of the corpus. What is
# left is the closure of the einops nodes plus the high-frequency tensor
# literacy under them, which is why the floor equals the count exactly: this
# graph is meant to stop shrinking. The retired pages are in
# This-Directory-Only/archive/retired-content-2026-08-30/ and their drills in
# Local_Deployed_Shared/pipeline/retired_question_ids.json.
_MIN_KCS = 37
# 380 -> 370 on 2026-07-28. The retirements cost exactly 7 tagged questions
# (six numpy.structured-dtypes drills plus q65), leaving 373. An earlier edit
# in the same pass dropped this to 343, which was 30 questions of unexplained
# slack -- a floor that loose would let a quarter of a lesson fall out of the
# ITS without failing. 370 keeps the small headroom a floor needs and nothing
# more.
# 370 -> 290 on 2026-08-30 with the same cut: 218 drills went with their
# concepts, leaving 293 tagged. Three questions of headroom, for the same
# reason as above -- the floor is here to catch a lesson falling out silently,
# not to permit one.
_MIN_TAGGED_QUESTIONS = 290


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



# ── Coverage guards added 2026-09-06 ─────────────────────────────────────────
# Each of these caught (or would have caught) a real silent gap while chapter
# 0.0 landed: es-1 shipped with no notebook JSON, and nothing at all watched
# arena_exercise_kcs.json — a mistyped KC or a fn key absent from the notebook
# simply meant no Practice button, with no error anywhere.

def _bank_ids():
    with open(os.path.join(_DIR, '..', 'questions.json'), encoding='utf-8') as fh:
        return {int(q['id']) for q in json.load(fh)}


def check_the_exercise_map_is_servable():
    """Every entry in arena_exercise_kcs.json must resolve end to end: the KC
    exists, every variant/original is a bank question whose q-matrix targets
    that KC, the notebook JSON exists, and the key (an ARENA `def fn` or a
    `(N)` heading tag) is present in that notebook's cells — otherwise
    exercise-session.js finds nothing to decorate and the button is silently
    absent."""
    import re
    registry = _read('kc_registry.json')
    qmatrix = _read('qmatrix_tags.json')
    ids = {kc['id'] for kc in registry['kcs']}
    bank = _bank_ids()
    exmap = _read('arena_exercise_kcs.json')
    problems = []
    for nb, table in exmap.items():
        if nb.startswith('_'):
            continue
        nb_path = os.path.join(_DIR, 'notebooks', f'arena-{nb}.json')
        if not os.path.exists(nb_path):
            problems.append(f'{nb}: notebooks/arena-{nb}.json missing')
            continue
        with open(nb_path, encoding='utf-8') as fh:
            cells = json.load(fh)['cells']
        src = '\n'.join(c.get('src', '') for c in cells)
        defs = set(re.findall(r'^\s*def\s+([A-Za-z_]\w*)\s*\(', src, re.M))
        tags = set(re.findall(r'^\s*(?:#+\s*)?(\(\w{1,3}\))\s', src, re.M))
        for key, ex in table.items():
            where = f'{nb}/{key}'
            if ex.get('kc') not in ids:
                problems.append(f'{where}: kc {ex.get("kc")!r} not in kc_registry.json')
            if key not in defs and key not in tags:
                problems.append(f'{where}: key not found as `def {key}(` or heading tag in arena-{nb}.json')
            qids = list(ex.get('variants') or [])
            if ex.get('original') is not None:
                qids.append(ex['original'])
            if not ex.get('variants'):
                problems.append(f'{where}: no variants — the planner has nothing to serve')
            for qid in qids:
                if qid not in bank:
                    problems.append(f'{where}: question {qid} not in questions.json')
                    continue
                targets = (qmatrix.get(str(qid)) or {}).get('target_kcs') or []
                if ex.get('kc') not in targets:
                    problems.append(f'{where}: q{qid} targets {targets} not {ex.get("kc")!r}')
    assert not problems, 'arena_exercise_kcs.json is not servable:\n  ' + '\n  '.join(problems[:20])


def check_the_glossary_points_at_live_kcs():
    """glossary.js hovers link a term to a KC and a lesson; a stale kc id
    renders a link that opens nothing."""
    import re
    ids = {kc['id'] for kc in _read('kc_registry.json')['kcs']}
    with open(os.path.join(_DIR, 'glossary.js'), encoding='utf-8') as fh:
        js = fh.read()
    kcs = set(re.findall(r'\bkc:\s*"([^"]+)"', js))
    lesson_keys = set(re.findall(r'^\s*"([a-z]+\.[a-z0-9-]+)":\s*\[', js, re.M))
    unknown = sorted((kcs | lesson_keys) - ids)
    assert kcs, 'glossary.js defines no kc: entries'
    assert not unknown, f'glossary.js references KCs missing from kc_registry.json: {unknown}'


def check_every_lesson_has_a_notebook():
    """Every lesson in kc_registry.json must be in lessons/colab_notebooks.json
    and have its web notebook under lessons/notebooks/ — es-1 shipped without
    one on 2026-09-06 and the Colab link 404ed."""
    registry = _read('kc_registry.json')
    manifest = _read('colab_notebooks.json')
    published = {l['id'] for l in manifest['lessons']}
    missing = []
    for lesson in registry['lessons']:
        lid = lesson['id']
        if lid not in published:
            missing.append(f'{lid}: not in colab_notebooks.json (run scripts/generate_colab_notebooks.py)')
        if not os.path.exists(os.path.join(_DIR, 'notebooks', f'{lid}.json')):
            missing.append(f'{lid}: notebooks/{lid}.json missing (run scripts/compile_web_notebooks.py)')
    assert not missing, 'lessons without a notebook:\n  ' + '\n  '.join(missing)


def check_every_kc_is_teachable():
    """Every KC needs a lesson it belongs to, at least one KP page compiled
    into lessons_structured.json, and at least one bank question whose
    q-matrix targets it. A KC failing any of these is a node the graph draws
    and the ITS can never serve."""
    registry = _read('kc_registry.json')
    qmatrix = _read('qmatrix_tags.json')
    lessons = {l['id'] for l in registry['lessons']}
    bank = _bank_ids()
    targeted = {}
    for qid, tags in qmatrix.items():
        if int(qid) in bank:
            for kc in tags.get('target_kcs') or []:
                targeted[kc] = targeted.get(kc, 0) + 1
    # Walk the compiled KP records themselves — a KC id merely MENTIONED
    # elsewhere (a supporting_kcs reference, a note) must not count as a page.
    compiled = {
        (kp.get('kc'), lesson.get('id'))
        for lesson in _read('lessons_structured.json')['lessons']
        for kp in lesson.get('kps') or []
    }
    problems = []
    for kc in registry['kcs']:
        kid = kc['id']
        if kc.get('lesson') not in lessons:
            problems.append(f'{kid}: lesson {kc.get("lesson")!r} not in registry lessons[]')
        if (kid, kc.get('lesson')) not in compiled:
            problems.append(f'{kid}: no KP compiled under lesson {kc.get("lesson")!r} in lessons_structured.json')
        if not targeted.get(kid):
            problems.append(f'{kid}: no bank question targets it — unservable')
    assert not problems, 'KCs the tutor cannot teach:\n  ' + '\n  '.join(problems)



# ── The three standing content guards ─────────
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
    checks = [check_imports, check_public_api, check_invariants,
              check_the_exercise_map_is_servable,
              check_the_glossary_points_at_live_kcs,
              check_every_lesson_has_a_notebook,
              check_every_kc_is_teachable] + _CONTENT_GUARDS
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
